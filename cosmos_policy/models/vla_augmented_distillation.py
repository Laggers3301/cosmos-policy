"""
Post-Hoc Adapter-Augmented Knowledge Distillation Modules.

Implements:
  - Adapter: MLP-based projection from Teacher hidden dim to Student hidden dim
  - DynamicRouter: Top-K routing mechanism over N specialized expert adapters
  - AdapterBank: Collection of 1 generalized + N specialized adapters with routing

Architecture reference:
  Teacher (Dream Zero CausalWanModel): dim=5120, fused context [B, 769, 5120]
  Student (Cosmos Policy WanModel):    dim=2048, cross-attn context [B, L, 2048]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdapterConfig:
    teacher_hidden_dim: int = 5120
    student_hidden_dim: int = 2048
    adapter_bottleneck_dim: int = 1024
    adapter_dropout: float = 0.1
    num_adapter_output_tokens: int = 16


class Adapter(nn.Module):
    """
    MLP-based adapter that projects Teacher hidden representations to
    the Student VLA's hidden dimension, with an optional bottleneck and
    sequence-length compression via a learnable query mechanism.

    Input:  H_T  [B, L_teacher, teacher_hidden_dim]   (e.g. [B, 769, 5120])
    Output: C    [B, num_output_tokens, student_hidden_dim]  (e.g. [B, 16, 2048])
    """

    def __init__(self, config: AdapterConfig):
        super().__init__()
        t_dim = config.teacher_hidden_dim     # 5120
        s_dim = config.student_hidden_dim     # 2048
        b_dim = config.adapter_bottleneck_dim  # 1024
        n_out = config.num_adapter_output_tokens  # 16

        self.down_proj = nn.Linear(t_dim, b_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(b_dim, s_dim)
        self.layer_norm = nn.LayerNorm(s_dim)
        self.dropout = nn.Dropout(config.adapter_dropout)

        # Learnable query tokens that cross-attend into the projected teacher
        # representations to compress the variable-length teacher sequence into
        # a fixed number of output tokens.
        self.query_tokens = nn.Parameter(
            torch.randn(1, n_out, s_dim) * 0.02
        )  # [1, num_output_tokens, student_hidden_dim]

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=s_dim,
            num_heads=8,
            dropout=config.adapter_dropout,
            batch_first=True,
        )
        self.post_attn_norm = nn.LayerNorm(s_dim)

    def forward(self, h_teacher: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_teacher: [B, L_teacher, teacher_hidden_dim]
        Returns:
            context:   [B, num_output_tokens, student_hidden_dim]
        """
        B = h_teacher.shape[0]

        # Project teacher features through bottleneck MLP
        # [B, L_teacher, teacher_hidden_dim] -> [B, L_teacher, student_hidden_dim]
        h = self.down_proj(h_teacher)   # [B, L_teacher, bottleneck_dim]
        h = self.act(h)                 # [B, L_teacher, bottleneck_dim]
        h = self.up_proj(h)             # [B, L_teacher, student_hidden_dim]
        h = self.layer_norm(h)          # [B, L_teacher, student_hidden_dim]
        h = self.dropout(h)             # [B, L_teacher, student_hidden_dim]

        # Cross-attend from learnable queries to compressed teacher features
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_output_tokens, student_hidden_dim]
        context, _ = self.cross_attn(
            query=queries,  # [B, num_output_tokens, student_hidden_dim]
            key=h,          # [B, L_teacher, student_hidden_dim]
            value=h,        # [B, L_teacher, student_hidden_dim]
        )  # context: [B, num_output_tokens, student_hidden_dim]

        context = self.post_attn_norm(context + queries)  # [B, num_output_tokens, student_hidden_dim]
        return context


class DynamicRouter(nn.Module):
    """
    Routing mechanism that produces Top-K selection probabilities over
    N specialized expert adapters, given a pooled multimodal representation.

    Uses a lightweight gating network on the pooled Teacher features.
    Includes noise injection during training for better exploration.

    Input:  h_pooled  [B, teacher_hidden_dim]
    Output: (top_k_weights, top_k_indices) each [B, top_k]
            expert_probs [B, num_experts] (for load-balancing loss)
    """

    def __init__(
        self,
        teacher_hidden_dim: int = 5120,
        num_experts: int = 4,
        top_k: int = 2,
        gating_hidden_dim: int = 512,
    ):
        super().__init__()
        assert top_k <= num_experts, f"top_k ({top_k}) must be <= num_experts ({num_experts})"

        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Sequential(
            nn.Linear(teacher_hidden_dim, gating_hidden_dim),
            nn.GELU(),
            nn.Linear(gating_hidden_dim, num_experts),
        )
        self.noise_weight = nn.Parameter(torch.zeros(1))

    def forward(
        self, h_pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_pooled: [B, teacher_hidden_dim]  (mean-pooled teacher features)
        Returns:
            top_k_weights:  [B, top_k]       normalized routing weights for selected experts
            top_k_indices:  [B, top_k]       indices of selected experts
            expert_probs:   [B, num_experts] full softmax probabilities (for aux loss)
        """
        logits = self.gate(h_pooled)  # [B, num_experts]

        if self.training:
            noise = torch.randn_like(logits) * F.softplus(self.noise_weight)
            logits = logits + noise  # [B, num_experts]

        expert_probs = F.softmax(logits, dim=-1)  # [B, num_experts]

        top_k_weights, top_k_indices = torch.topk(
            expert_probs, self.top_k, dim=-1
        )  # [B, top_k], [B, top_k]

        # Renormalize top-k weights to sum to 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        # [B, top_k]

        return top_k_weights, top_k_indices, expert_probs


class AdapterBank(nn.Module):
    """
    Full adapter bank consisting of:
      1) One Generalized Adapter (A_G) - always active
      2) N Specialized Adapters [A_S1, ..., A_Sn] - selected via dynamic routing
      3) A DynamicRouter for Top-K expert selection

    Forward pass:
      H_T -> A_G -> C_G                          [B, num_tokens, student_dim]
      H_T -> [A_S1..A_Sn] -> [C_S1..C_Sn]       each [B, num_tokens, student_dim]
      Pooled(H_T) -> Router -> Top-K weights/indices
      C_Agg = Concat(C_G, sum_topk(w_i * C_Si))  [B, 2*num_tokens, student_dim]
    """

    def __init__(
        self,
        teacher_hidden_dim: int = 5120,
        student_hidden_dim: int = 2048,
        adapter_bottleneck_dim: int = 1024,
        adapter_dropout: float = 0.1,
        num_adapter_output_tokens: int = 16,
        num_specialized_experts: int = 4,
        top_k: int = 2,
        gating_hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_specialized_experts = num_specialized_experts
        self.top_k = top_k

        adapter_cfg = AdapterConfig(
            teacher_hidden_dim=teacher_hidden_dim,
            student_hidden_dim=student_hidden_dim,
            adapter_bottleneck_dim=adapter_bottleneck_dim,
            adapter_dropout=adapter_dropout,
            num_adapter_output_tokens=num_adapter_output_tokens,
        )

        self.generalized_adapter = Adapter(adapter_cfg)

        self.specialized_adapters = nn.ModuleList(
            [Adapter(adapter_cfg) for _ in range(num_specialized_experts)]
        )

        self.router = DynamicRouter(
            teacher_hidden_dim=teacher_hidden_dim,
            num_experts=num_specialized_experts,
            top_k=top_k,
            gating_hidden_dim=gating_hidden_dim,
        )

    def forward(
        self, h_teacher: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_teacher: [B, L_teacher, teacher_hidden_dim]
                       Fused multimodal hidden states from the Teacher model.
        Returns:
            c_agg:        [B, 2 * num_adapter_output_tokens, student_hidden_dim]
                          Aggregated context (generalized + weighted specialized).
            expert_probs: [B, num_experts]
                          Full routing probabilities for the load-balancing loss.
        """
        B = h_teacher.shape[0]

        # --- Generalized Adapter ---
        c_g = self.generalized_adapter(h_teacher)  # [B, num_tokens, student_dim]

        # --- Specialized Adapters (all computed, but only top-k used) ---
        # Stack all specialized outputs for efficient gathering
        c_specialists = torch.stack(
            [adapter(h_teacher) for adapter in self.specialized_adapters], dim=1
        )  # [B, num_experts, num_tokens, student_dim]

        # --- Dynamic Routing ---
        h_pooled = h_teacher.mean(dim=1)  # [B, teacher_hidden_dim]
        top_k_weights, top_k_indices, expert_probs = self.router(h_pooled)
        # top_k_weights: [B, top_k], top_k_indices: [B, top_k], expert_probs: [B, num_experts]

        # Gather the top-k expert outputs
        num_tokens = c_g.shape[1]
        student_dim = c_g.shape[2]

        # Expand indices to gather from the specialist tensor
        gather_idx = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(
            B, self.top_k, num_tokens, student_dim
        )  # [B, top_k, num_tokens, student_dim]

        selected_c_s = torch.gather(
            c_specialists, dim=1, index=gather_idx
        )  # [B, top_k, num_tokens, student_dim]

        # Weight and sum the selected expert outputs
        weighted_c_s = (
            selected_c_s * top_k_weights.unsqueeze(-1).unsqueeze(-1)
        )  # [B, top_k, num_tokens, student_dim]
        c_s_combined = weighted_c_s.sum(dim=1)  # [B, num_tokens, student_dim]

        # --- Aggregation ---
        c_agg = torch.cat(
            [c_g, c_s_combined], dim=1
        )  # [B, 2 * num_tokens, student_dim]

        return c_agg, expert_probs
