"""
Loss functions for adapter-augmented MoE knowledge distillation.

Implements:
  1. ActionChunkLoss:  L1 / MSE loss on predicted vs target action chunks
                       (wraps the Cosmos Policy EDM diffusion loss)
  2. LoadBalancingLoss: Auxiliary loss to prevent expert collapse in the
                        dynamic routing mechanism
  3. DistillationLoss:  Combined total loss with configurable weighting:
                        L_total = L_action + alpha * L_load_balance
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillationLossConfig:
    load_balance_weight: float = 0.01
    action_loss_type: str = "l1"
    num_experts: int = 4


class LoadBalancingLoss(nn.Module):
    """
    Auxiliary load-balancing loss for the Mixture-of-Experts routing mechanism.

    Encourages uniform expert utilization by penalizing the variance of
    routing probabilities across the batch.  Follows the Switch Transformer
    formulation (Fedus et al., 2022):

        L_balance = num_experts * sum_i( f_i * P_i )

    where:
        f_i = fraction of tokens routed to expert i (hard assignment)
        P_i = mean routing probability for expert i across the batch

    When all experts are used equally, L_balance = 1.0 (its minimum).
    Expert collapse drives L_balance >> 1.

    Input:  expert_probs [B, num_experts]  (full softmax probabilities from router)
    Output: scalar loss
    """

    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expert_probs: [B, num_experts]  softmax probabilities from DynamicRouter
        Returns:
            loss: scalar  load-balancing loss
        """
        # f_i: fraction of the batch where expert i has the highest routing probability
        # [B, num_experts] -> argmax -> one-hot -> mean over batch -> [num_experts]
        top_expert = expert_probs.argmax(dim=-1)  # [B]
        expert_counts = torch.zeros(
            self.num_experts, device=expert_probs.device, dtype=expert_probs.dtype
        )
        expert_counts.scatter_add_(
            0, top_expert, torch.ones_like(top_expert, dtype=expert_probs.dtype)
        )
        f_i = expert_counts / expert_probs.shape[0]  # [num_experts]

        # P_i: mean routing probability for each expert across the batch
        p_i = expert_probs.mean(dim=0)  # [num_experts]

        # Switch Transformer load-balancing loss
        loss = self.num_experts * (f_i * p_i).sum()  # scalar
        return loss


class ActionChunkLoss(nn.Module):
    """
    Direct action chunk prediction loss (L1 or MSE).

    This is used when computing an explicit action loss on extracted
    action chunks (e.g., for auxiliary supervision or when bypassing
    the full diffusion loss).  The primary training signal from the
    Cosmos Policy student already uses EDM diffusion loss; this module
    provides an optional additional term.

    Input:  pred_actions   [B, chunk_size, action_dim]
            target_actions [B, chunk_size, action_dim]
    Output: scalar loss
    """

    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        assert loss_type in ("l1", "mse"), f"Unsupported loss_type: {loss_type}"
        self.loss_type = loss_type

    def forward(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_actions:   [B, chunk_size, action_dim]
            target_actions: [B, chunk_size, action_dim]
            mask:           [B, chunk_size] optional per-timestep mask
        Returns:
            loss: scalar
        """
        if self.loss_type == "l1":
            per_elem_loss = F.l1_loss(
                pred_actions, target_actions, reduction="none"
            )  # [B, chunk_size, action_dim]
        else:
            per_elem_loss = F.mse_loss(
                pred_actions, target_actions, reduction="none"
            )  # [B, chunk_size, action_dim]

        if mask is not None:
            # mask: [B, chunk_size] -> [B, chunk_size, 1]
            per_elem_loss = per_elem_loss * mask.unsqueeze(-1)
            return per_elem_loss.sum() / (mask.sum() * pred_actions.shape[-1] + 1e-8)

        return per_elem_loss.mean()


class DistillationLoss(nn.Module):
    """
    Combined loss for the MoE VLA distillation framework.

    L_total = L_student + alpha * L_load_balance

    where:
      L_student:       The student's native EDM diffusion loss (from training_step).
                       This is the primary action prediction loss operating in
                       the latent diffusion space.
      L_load_balance:  Auxiliary load-balancing loss on the routing probabilities.

    The student's diffusion loss already encodes the action prediction objective
    (since actions are injected into latent frames and the diffusion model
    learns to denoise them).  The load-balancing loss prevents expert collapse.
    """

    def __init__(self, config: DistillationLossConfig):
        super().__init__()
        self.config = config
        self.load_balance_loss = LoadBalancingLoss(num_experts=config.num_experts)
        self.action_chunk_loss = ActionChunkLoss(loss_type=config.action_loss_type)

    def forward(
        self,
        student_loss: torch.Tensor,
        expert_probs: torch.Tensor,
        pred_actions: torch.Tensor | None = None,
        target_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute the total distillation loss.

        Args:
            student_loss:   scalar, the student's EDM diffusion loss
            expert_probs:   [B, num_experts], routing probabilities
            pred_actions:   [B, chunk_size, action_dim] (optional, for aux action loss)
            target_actions: [B, chunk_size, action_dim] (optional, for aux action loss)

        Returns:
            total_loss: scalar combined loss
            loss_dict:  breakdown of individual loss components for logging
        """
        lb_loss = self.load_balance_loss(expert_probs)  # scalar

        total_loss = student_loss + self.config.load_balance_weight * lb_loss

        if not torch.isfinite(student_loss):
            raise ValueError(
                f"student_loss is not finite (nan/inf): {student_loss.item()}. "
                "Common causes: (1) Teacher (Dream Zero) output has NaN — try teacher in float32 or check teacher checkpoint; "
                "(2) Injected adapter context c_agg has NaN; (3) Student diffusion sigma underflow or bad condition."
            )
        if not torch.isfinite(lb_loss):
            raise ValueError(
                f"load_balance_loss is not finite: {lb_loss.item()}. "
                "Check expert_probs (router logits / teacher features)."
            )

        loss_dict = {
            "total_loss": total_loss.detach(),
            "student_edm_loss": student_loss.detach(),
            "load_balance_loss": lb_loss.detach(),
            "load_balance_weight": torch.tensor(self.config.load_balance_weight),
        }

        # Optional auxiliary action chunk loss (if action predictions are available)
        if pred_actions is not None and target_actions is not None:
            aux_action_loss = self.action_chunk_loss(pred_actions, target_actions)
            total_loss = total_loss + aux_action_loss
            loss_dict["aux_action_loss"] = aux_action_loss.detach()
            loss_dict["total_loss"] = total_loss.detach()

        return total_loss, loss_dict
