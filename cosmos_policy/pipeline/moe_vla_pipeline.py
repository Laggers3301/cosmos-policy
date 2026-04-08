"""
MoE VLA Distillation Pipeline.

Wraps a frozen Dream Zero Teacher and a trainable Cosmos Policy Student,
injecting adapter-augmented teacher contexts into the student's cross-attention
via a forward hook on the student's text_embedding layer.

Architecture:
  Teacher (Dream Zero):  Full forward pass through 40-layer CausalWanModel
                         → last transformer block hidden states
                         H_T: [B, L_total, 5120]
                         (L_total = video_patches + action/state tokens)

  AdapterBank:           H_T -> (Generalized + TopK Specialized) -> C_Agg [B, 2K, 2048]

  Student (Cosmos Policy): Original cross-attn context: text_embedding(T5_emb) -> [B, L, 2048]
                           Augmented context: cat(C_Agg, text_emb) -> [B, 2K+L, 2048]
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmos_policy.models.vla_augmented_distillation import AdapterBank

# Dream Zero Wan2.1-I2V-14B-480P 使用的 480p 分辨率（宽×高），会得到 frame_seqlen=1560，与 DiT 内 880 不一致会触发 assert
# TEACHER_480P_WIDTH = 832
# TEACHER_480P_HEIGHT = 480

# 与 DiT 内 frame_seqlen=880 匹配的整图分辨率：2×2 拼图后 352×640（VAE 后 44×80 latent → 880 token/帧）
TEACHER_FRAME_SEQLEN_880_HEIGHT = 352
TEACHER_FRAME_SEQLEN_880_WIDTH = 640
# 单视角 tile 尺寸（Dream Zero 配置中 VideoResize 的 height/width），拼成 2×2 后为上面整图
TEACHER_SINGLE_VIEW_HEIGHT = 176
TEACHER_SINGLE_VIEW_WIDTH = 320

logger = logging.getLogger(__name__)


@dataclass
class MoEVLAPipelineConfig:
    # Teacher (Dream Zero) dimensions
    teacher_hidden_dim: int = 5120
    teacher_text_dim: int = 4096
    teacher_clip_dim: int = 1280

    # Student (Cosmos Policy) dimensions
    student_hidden_dim: int = 2048

    # Adapter configuration
    adapter_bottleneck_dim: int = 1024
    adapter_dropout: float = 0.1
    num_adapter_output_tokens: int = 16
    num_specialized_experts: int = 4
    top_k: int = 2
    gating_hidden_dim: int = 512

    # Loss weighting
    load_balance_loss_weight: float = 0.01

    # 为 True 时 student_loss 转为 float32 再参与 backward（前向仍 bf16，避免 SDPA 无 float32 kernel）
    use_action_fp32: bool = False

    # Teacher 取第几层的输出作为 H_T（1-based）。默认 40 即最后一层；设为 14 则只保留前 14 层并取第 14 层输出，显存更小。
    teacher_layer_index: int = 40


class _AttrDict(dict):
    """Dict subclass with attribute-style access (BatchFeature drop-in)."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class TeacherFeatureExtractor(nn.Module):
    """
    Extracts deep multimodal hidden states H_T from Dream Zero's transformer
    at a given layer (default: last block) via a forward hook on CausalWanAttentionBlock.

    When layer_index < num_layers, the backbone is truncated to the first layer_index
    blocks so that layers (layer_index+1)..40 are never run and not kept in memory,
    reducing both compute and GPU memory (parameter + activation).

    Produces:
        H_T: [B, L_total, 5120]
             L_total = seq_len (video patches) + action_register_length
             The exact value varies per batch depending on spatial resolution,
             number of frames, and action/state token counts.
    """

    def __init__(self, teacher_vla: nn.Module, layer_index: int = 40):
        super().__init__()
        self.teacher_vla = teacher_vla
        self.teacher_vla.eval()
        for p in self.teacher_vla.parameters():
            p.requires_grad = False

        self._base_model = self._resolve_base_model()
        num_blocks = len(self._base_model.blocks)
        # 只用到第 layer_index 层：截断为前 layer_index 层，后续层不加载在计算图中，显存更小
        if layer_index < num_blocks:
            self._base_model.blocks = nn.ModuleList(self._base_model.blocks[:layer_index])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "TeacherFeatureExtractor: truncated teacher to first %d layers "
                "(dropped %d layers to save memory).",
                layer_index,
                num_blocks - layer_index,
            )
            num_blocks = layer_index

        self._last_block_output: Optional[torch.Tensor] = None
        self._hook_handle = self._base_model.blocks[-1].register_forward_hook(
            self._capture_last_block_hook
        )
        logger.info(
            "TeacherFeatureExtractor: registered hook on transformer "
            "block (index %d of %d)",
            len(self._base_model.blocks) - 1,
            len(self._base_model.blocks),
        )

    def _resolve_base_model(self):
        """Navigate through PEFT wrapper to get CausalWanModel."""
        model = self.teacher_vla.action_head.model
        if hasattr(model, "base_model"):
            return model.base_model.model
        return model

    def _capture_last_block_hook(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        """
        Forward hook on the last CausalWanAttentionBlock.

        Block.forward() returns ``(x, kv_cache)``.
        We capture ``x`` and detach to sever any gradient flow.
        """
        x = output[0] if isinstance(output, tuple) else output
        self._last_block_output = x.detach()  # [B, L_total, 5120]

    def forward(self, teacher_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run the teacher's full forward pass and return the last block output.

        **Teacher 输入说明（images 帧数约束）**
        - Dream Zero 的 ``encode_image(image, num_frames, ...)`` 里对 mask 做
          ``msk = [第一帧×4, 其余帧]`` 再 ``view(..., msk.shape[1]//4, 4, ...)``，
          要求 ``msk.shape[1] = 4 + (num_frames - 1)`` 能被 4 整除，
          即 **num_frames 必须为 1, 5, 9, 13, ...**（(T+3) % 4 == 0）。
        - 训练分支里 ``num_frames`` 来自 ``videos.shape[2]``（即 ``images`` 的时间维 T），
          所以 **images 的时间维 T 必须为 1 或 5/9/13...**，传 2 帧会触发 view 报错。
        - **为何一开始会传 2 帧**：RoboCasa 数据集按「当前步 + 未来步」取两帧（左第三人称），
          与 Student 的 future 条件对齐，TEACHER_INPUT_SOURCE 里 teacher_raw_frames 即这两帧；
          当时未考虑 Dream Zero 对 T 的 1/5/9/13 假设。
        - **应该传几帧、什么图像**：
          - **推荐**：只传 **1 帧**（当前步观测），即 ``images [B, 1, H, W, C]``，
            满足约束且语义清晰；若数据侧当前是两帧，取 ``images[:, :1]`` 即可。
          - 若需保留多帧语义：可传 **5 帧**（如当前帧重复 5 次或从轨迹取 5 步），
            需在数据集/collate 里构造为 (B, 5, H, W, C)。

        **Gradient handling** — this method must NOT be called inside a
        ``torch.no_grad()`` context.  ``CausalWanModel._forward_train`` has::

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = checkpoint(create_custom_forward(block), x, **kw)
            else:
                x = block(x, **kw)          # ← returns (x, kv_cache) tuple

        The ``else`` branch assigns the raw ``(x, kv_cache)`` tuple to ``x``,
        which breaks subsequent blocks.  Keeping grad enabled ensures the
        ``checkpoint`` path is taken, where ``create_custom_forward`` correctly
        unpacks the tuple.  Because every teacher parameter has
        ``requires_grad=False`` and input data does not require grad, no
        computation graph is actually built.

        Args:
            teacher_inputs: Dream-Zero-formatted dict with at minimum::

                images              – [B, T, H, W, C]  video frames
                text                – [B, L]            token IDs
                text_attention_mask – [B, L]
                action              – [B, T_a, action_dim]  in [-1, 1]
                state               – [B, T_s, state_dim]
                embodiment_id       – [B]
                has_real_action     – bool or [B]
                action_mask         – [B, T_a]

        Returns:
            h_teacher: [B, L_total, 5120]  last transformer block output

        **Dream Zero 训练输入形式与源码对应**（对齐用）:
        - action_head 入口与 prepare_input:
          dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf_efficient_weighted.py
          - forward(backbone_output, action_input)  L599: data = action_input，L616: videos = data["images"]
          - prepare_input(batch)  L540-541: return BatchFeature(data=batch)，batch 原样传入
        - 期望的 action_input 键: images, text, text_attention_mask, action, state,
          embodiment_id, has_real_action, action_mask（与 BaseVLA.prepare_input 一致）
        - images 形状与 num_frames 来源:
          - L617-618: videos = data["images"]; rearrange(videos, "b t h w c -> b c t h w")
            即 images 为 [B, T, H, W, C]，T 为帧数
          - L638: _, _, num_frames, height, width = videos.shape  （num_frames 取自输入 T，非 config）
          - L643: encode_image(image, num_frames, height, width)
        - encode_image 对 T 的约束（L561-567）:
          msk 构造后做 view(..., msk.shape[1]//4, 4, ...)，要求 (T+3)%4==0，
          即 T 只能为 1, 5, 9, 13, 17, 21, ...
          源码: msk = concat([repeat_interleave(msk[:,0:1], 4), msk[:,1:]]); msk.view(..., msk.shape[1]//4, 4, ...)
        - Dream Zero 自身训练时 batch["images"] 的来源:
          dreamzero/groot/vla/model/dreamzero/transform/dreamzero_cotrain.py
          - apply_single: _prepare_video(data) 得到 [V, T, C, H, W]，_apply_vlm_processing 里
            rearrange(images, "v t c h w -> (t v) h w c") 得到 (T*V, H, W, C)
          - collate 后 batch["images"] 为 [B, T*V, H, W, C]；单视角即 [B, T, H, W, C]，T 由数据集 data["video"] 的时间维决定，通常与 config.num_frames（如 21）一致。

        **Dream Zero 训练输入「内容」要求（帧数/步数）与源码**:
        - 视频帧数 T：训练时通常为 config.num_frames（如 21），且 (T+3)%4==0。
          来源：数据集 data["video"] 时间维；transform 见上。
        - state 步数 T_s、action 步数 T_a 与 T 的定量关系（action_head 内由输入 T 推导）:
          dreamzero/groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf_efficient_weighted.py
          - L638: num_frames = videos.shape[2]（即输入的 T）
          - L681-683: 断言
            actions.shape[1] / (noise.shape[1]-1) == num_action_per_block // num_frame_per_block  （默认 32）
            (noise.shape[1]-1) / state_features.shape[1] == num_frame_per_block // num_state_per_block  （默认 1）
          - noise 时间维 = latent_T = 1 + (T-1)//4，故 num_blocks = (noise.shape[1]-1) = (T-1)//4。
          因此:
          - state 时间维 state_steps = num_blocks = (T-1)//4  （默认 num_state_per_block=1, num_frame_per_block=1）
          - action 时间维 action_steps = num_blocks * (num_action_per_block // num_frame_per_block) = (T-1)//4 * 32
          出处：wan_video_dit_action_casual_chunk.py L902-904 同式（action_horizon = num_image_blocks * num_action_per_block, state_horizon = num_image_blocks * num_state_per_block）。
        - 示例（T=21）：state_steps=5, action_steps=160；T=5 时 state_steps=1, action_steps=32。

        **Dream Zero 中 config 与输入处理的对应关系**:
        - 是的，输入处理流程与 config 对应，但对应方式是「约定一致」而非「自动读取」：
          - 数据侧：DreamTransform（dreamzero_cotrain.py L195-196）的 state_horizon、action_horizon 是
            **transform 的构造参数**，由训练脚本/配置传入；数据集提供的 data["video"] 时间维 T 与
            transform 的 state/action 长度需满足模型那套公式（T_s=(T-1)//4, T_a=(T-1)//4*32）。
          - 模型侧：action_head 训练时 num_frames 取自**输入** videos.shape[2]；assert 里的
            num_action_per_block、num_state_per_block 来自**模型/DiT 的 config**。
        - **不会**随 model config 自动适配：若只改 model 的 num_frames（或 block 配置），必须**同时**改
          transform 的 state_horizon/action_horizon 以及数据集提供的 T，否则 forward 断言会报错。
        - 本 pipeline **会**自适应：通过 get_teacher_input_dims() 从**当前加载的 teacher** 的 config 与
          model 读出 num_frames、num_action_per_block、num_state_per_block 并算出 state_steps、action_steps，
          再对 teacher_inputs 做 pad/截断，因此换用不同 config 的 Dream Zero checkpoint 时无需改代码。
        """
        teacher_inputs = dict(teacher_inputs)
        image_text_only = teacher_inputs.pop("_teacher_image_text_only", False)

        num_frames, state_steps, action_steps = self.get_teacher_input_dims()
        ah = self.teacher_vla.action_head
        cfg = getattr(ah, "config", None) or {}
        def _get_cfg(key, default=None):
            return getattr(cfg, key, None) if hasattr(cfg, key) else cfg.get(key, default)
        max_state_dim = _get_cfg("max_state_dim")
        max_action_dim = _get_cfg("max_action_dim")
        if max_action_dim is None:
            max_action_dim = self.get_teacher_action_dim()
        if max_state_dim is None:
            max_state_dim = 64  # Dream Zero config default; state pad/trunc to this

        # --- Optional: teacher image+text only (no action/state from dataset) ---
        # Dream Zero action_head builds timestep_id from action time dim; 0 steps causes reshape
        # "shape '[1, -1, 2]' is invalid for input of size 1". So we pass dummy action/action_mask
        # with full action_steps shape (zeros + mask all False) so DiT runs without using action.
        if image_text_only:
            B = teacher_inputs["images"].shape[0]
            device = teacher_inputs["images"].device
            teacher_inputs["action"] = torch.zeros(B, action_steps, max_action_dim, dtype=torch.float32, device=device)
            teacher_inputs["action_mask"] = torch.zeros(B, action_steps, max_action_dim, dtype=torch.bool, device=device)
            teacher_inputs["state"] = torch.zeros(B, state_steps, max_state_dim, dtype=torch.float32, device=device)

        # --- RoboCasa (7D action, 9D state) -> DreamZero (28D action, 14D state, Euler pose) ---
        # Convert what can be converted (pos + rotation->euler + gripper), pad rest with 0 and set mask.
        # Skip when image_text_only: we already set empty action/state above.
        state = teacher_inputs.get("state")
        action = teacher_inputs.get("action")
        _TorchUtils = None
        try:
            from robocasa.utils.robomimic.robomimic_torch_utils import (
                axis_angle_to_matrix,
                matrix_to_euler_angles,
                quaternion_to_matrix,
            )
            _TorchUtils = True
        except Exception:
            pass
        EULER_CONVENTION = "XYZ"
        ROBOCASA_STATE_DIM = 9
        ROBOCASA_ACTION_DIM = 7
        DREAMZERO_STATE_DIM = 14
        DREAMZERO_ACTION_DIM = 28

        if _TorchUtils and state is not None and state.shape[-1] == ROBOCASA_STATE_DIM:
            # RoboCasa state 9D: eef_pos(3), eef_quat(4), gripper(1), extra(1) -> DreamZero 14D: cartesian(6), gripper(1), joint(7)
            # quat assumed (w,x,y,z) at state[..., 3:7]
            pos = state[..., 0:3]
            quat = state[..., 3:7]
            gripper = state[..., 7:8]
            mat = quaternion_to_matrix(quat)
            euler = matrix_to_euler_angles(mat, EULER_CONVENTION)
            joint_pad = torch.zeros(
                (*state.shape[:-1], 7),
                dtype=state.dtype,
                device=state.device,
            )
            state = torch.cat([pos, euler, gripper, joint_pad], dim=-1)
            teacher_inputs["state"] = state

        if _TorchUtils and action is not None and action.shape[-1] == ROBOCASA_ACTION_DIM:
            # RoboCasa action 7D: pos(3), axis_angle(3), gripper(1) -> DreamZero 28D (Euler cartesian_pos + rest 0)
            pos = action[..., 0:3]
            axis_angle = action[..., 3:6]
            gripper = action[..., 6:7]
            mat = axis_angle_to_matrix(axis_angle)
            euler = matrix_to_euler_angles(mat, EULER_CONVENTION)
            B, T = action.shape[0], action.shape[1]
            device, dtype = action.device, action.dtype
            cartesian_pos = torch.cat([pos, euler], dim=-1)
            cartesian_vel = torch.zeros(B, T, 6, dtype=dtype, device=device)
            gripper_vel = torch.zeros(B, T, 1, dtype=dtype, device=device)
            joint_pos = torch.zeros(B, T, 7, dtype=dtype, device=device)
            joint_vel = torch.zeros(B, T, 7, dtype=dtype, device=device)
            action = torch.cat(
                [
                    cartesian_pos,
                    cartesian_vel,
                    gripper,
                    gripper_vel,
                    joint_pos,
                    joint_vel,
                ],
                dim=-1,
            )
            teacher_inputs["action"] = action
            # action_mask: first 7 dims (converted) valid, rest (padded) invalid
            valid_dims = ROBOCASA_ACTION_DIM
            amask = torch.zeros(
                B, T, DREAMZERO_ACTION_DIM,
                dtype=torch.bool,
                device=device,
            )
            amask[..., :valid_dims] = True
            teacher_inputs["action_mask"] = amask

        # Align images to teacher's num_frames (must satisfy (T+3)%4==0)
        # 训练 vs 推理：Dream Zero 训练时数据通常为 33 帧；推理时 forward 可接受 1/4/9 帧输入，
        # 内部用 config.num_frames(33) 做 conditioning。我们这里走的是 training forward（无 kv_cache），
        # 故 num_frames 取自输入 T，只需满足 (T+3)%4==0，即 T=1,5,9,...,33 均可。
        # Dream Zero DiT 用实际帧数算 num_image_blocks；T_latent=1+(T-1)//4，num_image_blocks=(T_latent-1)//2。
        # 若 T=5 则 T_latent=2、num_image_blocks=0 → timestep_id 只有 1 元素 → reshape 报错。故任何路径下至少需要 9 帧（T_latent=3, num_image_blocks>=1）。
        TEACHER_MIN_FRAMES = 9  # (9+3)%4==0, T_latent=3, num_image_blocks>=1
        images = teacher_inputs.get("images")
        if images is not None and images.dim() >= 4:
            T = images.shape[1]
            if T != num_frames:
                if T < num_frames:
                    if image_text_only:
                        # 只传图像+文本时：至少 9 帧满足 num_image_blocks>=1，用 9 帧省显存（不扩到 33）
                        target_T = max(T, TEACHER_MIN_FRAMES)
                        if target_T > T:
                            if T == 1:
                                teacher_inputs["images"] = images.expand(-1, target_T, -1, -1, -1).contiguous()
                            else:
                                last = images[:, -1:].expand(-1, target_T - T, -1, -1, -1)
                                teacher_inputs["images"] = torch.cat([images, last], dim=1).contiguous()
                    elif T == 1:
                        # 1 帧：直接 repeat 到 num_frames（Dream Zero 支持 T=1）
                        teacher_inputs["images"] = images.expand(-1, num_frames, -1, -1, -1).contiguous()
                    elif T == 2:
                        # 2 帧 → 9 帧（满足 (T+3)%4==0 且 num_image_blocks>=1）：前 5 帧 frame0，后 4 帧 frame1；5 帧会致 num_image_blocks=0、timestep_id reshape 报错
                        frame0 = images[:, 0:1]
                        frame1 = images[:, 1:2]
                        teacher_inputs["images"] = torch.cat([
                            frame0.expand(-1, 5, -1, -1, -1),
                            frame1.expand(-1, 4, -1, -1, -1),
                        ], dim=1).contiguous()
                    else:
                        last = images[:, -1:].expand(-1, num_frames - T, -1, -1, -1)
                        teacher_inputs["images"] = torch.cat([images, last], dim=1).contiguous()
                else:
                    teacher_inputs["images"] = images[:, :num_frames].contiguous()

            images = teacher_inputs["images"]
            if images.dim() == 5:
                B, T, h, w, C = images.shape
                if h != TEACHER_FRAME_SEQLEN_880_HEIGHT or w != TEACHER_FRAME_SEQLEN_880_WIDTH:
                    # 1) 直接缩放到 176×320，与 cv2.resize(img, (320, 176)) 行为一致（不保比例，目标尺寸即输出尺寸）
                    # F.interpolate(mode="bilinear") 不支持 uint8，先转 float 再插值，再转回 uint8
                    target_h, target_w = TEACHER_SINGLE_VIEW_HEIGHT, TEACHER_SINGLE_VIEW_WIDTH
                    # [B, T, H, W, C] -> [B*T, C, H, W]
                    flat = images.permute(0, 1, 4, 2, 3).reshape(B * T, C, h, w)
                    if flat.dtype == torch.uint8:
                        flat = flat.to(torch.float32)
                    flat = F.interpolate(
                        flat,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    if images.dtype == torch.uint8:
                        flat = flat.clamp(0, 255).round().to(torch.uint8)
                    # -> [B, T, 176, 320, C]
                    single_view = flat.reshape(B, T, C, target_h, target_w).permute(0, 1, 3, 4, 2).contiguous()

                    # 2) 拼成 2×2：整图 352×640，左上/右上/左下放同一张 single_view，右下留黑
                    grid_h, grid_w = TEACHER_FRAME_SEQLEN_880_HEIGHT, TEACHER_FRAME_SEQLEN_880_WIDTH
                    out = torch.zeros(
                        B, T, grid_h, grid_w, C,
                        dtype=single_view.dtype,
                        device=single_view.device,
                    )
                    out[:, :, :target_h, :target_w, :] = single_view   # 左上
                    out[:, :, :target_h, target_w:, :] = single_view   # 右上
                    out[:, :, target_h:, :target_w, :] = single_view   # 左下
                    # 右下 [target_h:, target_w:] 保持为 0
                    teacher_inputs["images"] = out

        # 按实际送入的帧数重算 state_steps/action_steps，否则 9 帧时仍用 33 帧的 96 步会触发 Dream Zero assert
        T_final = teacher_inputs["images"].shape[1]
        state_steps, action_steps = self.get_teacher_input_dims_for_frames(T_final)

        if image_text_only:
            # image_text_only 时前面按 config 设了 dummy，需按实际帧数对应的步数重设
            B = teacher_inputs["images"].shape[0]
            dev = teacher_inputs["images"].device
            teacher_inputs["action"] = torch.zeros(B, action_steps, max_action_dim, dtype=torch.float32, device=dev)
            teacher_inputs["action_mask"] = torch.zeros(B, action_steps, max_action_dim, dtype=torch.bool, device=dev)
            teacher_inputs["state"] = torch.zeros(B, state_steps, max_state_dim, dtype=torch.float32, device=dev)

        # Align state to teacher's state_steps: [B, T_s_in, state_dim] -> [B, state_steps, state_dim]
        state = teacher_inputs.get("state")
        if state is not None and state_steps > 0:
            B, T_s_in, state_dim = state.shape[0], state.shape[1], state.shape[2]
            if T_s_in != state_steps:
                if T_s_in < state_steps:
                    last_s = state[:, -1:].expand(-1, state_steps - T_s_in, -1)
                    state = torch.cat([state, last_s], dim=1).contiguous()
                else:
                    state = state[:, :state_steps].contiguous()
            if max_state_dim is not None and state.shape[2] != max_state_dim:
                if state.shape[2] < max_state_dim:
                    pad = torch.zeros(B, state.shape[1], max_state_dim - state.shape[2], dtype=state.dtype, device=state.device)
                    state = torch.cat([state, pad], dim=2).contiguous()
                else:
                    state = state[:, :, :max_state_dim].contiguous()
            teacher_inputs["state"] = state

        # Align action and action_mask to teacher's action_steps (skip when image_text_only: keep empty action)
        # 已注释：原写入 /workspace/debug.txt 并设置 COSMOS_DEBUG_SHAPE_FILE
        # _debug_path = os.environ.get("COSMOS_DEBUG_SHAPE_FILE", "/workspace/debug.txt")
        # os.environ["COSMOS_DEBUG_SHAPE_FILE"] = _debug_path
        action = teacher_inputs.get("action")
        if not image_text_only and action is not None and action_steps > 0 and action.shape[1] > 0:
            B, T_a_in, action_dim = action.shape[0], action.shape[1], action.shape[2]
            # with open(_debug_path, "a") as _f: ...
            if T_a_in != action_steps:
                if T_a_in < action_steps:
                    last_a = action[:, -1:].expand(-1, action_steps - T_a_in, -1)
                    action = torch.cat([action, last_a], dim=1).contiguous()
                else:
                    action = action[:, :action_steps].contiguous()
            # with open(_debug_path, "a") as _f: ...
            if max_action_dim is not None and action.shape[2] != max_action_dim:
                if action.shape[2] < max_action_dim:
                    pad = torch.zeros(B, action.shape[1], max_action_dim - action.shape[2], dtype=action.dtype, device=action.device)
                    action = torch.cat([action, pad], dim=2).contiguous()
                else:
                    action = action[:, :, :max_action_dim].contiguous()
            # with open(_debug_path, "a") as _f: ...
            teacher_inputs["action"] = action
            # Ensure [-1, 1] and no NaN/Inf for Dream Zero action_head assert
            a = teacher_inputs["action"]
            teacher_inputs["action"] = torch.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        amask = teacher_inputs.get("action_mask")
        if not image_text_only and amask is not None and action_steps > 0 and amask.shape[1] > 0:
            # with open(_debug_path, "a") as _f: ...
            T_a_in = amask.shape[1]
            if T_a_in != action_steps:
                if T_a_in < action_steps:
                    pad_shape = [amask.shape[0], action_steps - T_a_in]
                    if amask.dim() == 3:
                        pad_shape.append(amask.shape[2])
                    pad = torch.zeros(pad_shape, dtype=amask.dtype, device=amask.device)
                    amask = torch.cat([amask, pad], dim=1).contiguous()
                else:
                    amask = amask[:, :action_steps].contiguous()
            # with open(_debug_path, "a") as _f: ...
            # Dream Zero expects action_mask [B, T, max_action_dim]
            if amask.dim() == 2:
                amask = amask.unsqueeze(-1).expand(-1, -1, max_action_dim).contiguous()
            elif amask.shape[2] != max_action_dim:
                if amask.shape[2] < max_action_dim:
                    pad = torch.zeros(amask.shape[0], amask.shape[1], max_action_dim - amask.shape[2], dtype=amask.dtype, device=amask.device)
                    amask = torch.cat([amask, pad], dim=2).contiguous()
                else:
                    amask = amask[:, :, :max_action_dim].contiguous()
            # with open(_debug_path, "a") as _f: ...
            teacher_inputs["action_mask"] = amask

        # 已注释：原 Debug 写入 /workspace/debug.txt
        # def _shape(k): ...
        # with open(_debug_path, "a") as _f: ...
        # Force gradient_checkpointing on so the block loop takes the
        # create_custom_forward path that properly unpacks (x, kv_cache).
        orig_gc = getattr(self._base_model, "gradient_checkpointing", False)
        self._base_model.gradient_checkpointing = True

        try:
            self.teacher_vla.action_head(
                _AttrDict(),               # backbone_output (unused by action_head)
                _AttrDict(teacher_inputs),  # action_input
            )
        finally:
            self._base_model.gradient_checkpointing = orig_gc

        h_teacher = self._last_block_output
        self._last_block_output = None

        if h_teacher is None:
            raise RuntimeError(
                "Forward hook on the last transformer block did not fire.  "
                "Verify that teacher_inputs triggers "
                "CausalWanModel._forward_train."
            )

        return h_teacher  # [B, L_total, 5120]

    def get_teacher_input_dims(self) -> Tuple[int, int, int]:
        """
        从 Dream Zero teacher 的 config 与 model（DiT）推导期望的输入维度。
        DiT 内部用公式 action_horizon = num_image_blocks * num_action_per_block、
        state_horizon = num_image_blocks * num_state_per_block 与 roped_query 对齐，
        因此 state_steps/action_steps 必须按同一公式从 num_frames 与 model 的 block 配置推导，
        不能使用 config 里的 state_horizon/action_horizon（可能与 DiT 不一致，导致 L912 assert 失败）。
        Returns:
            num_frames: 视频帧数（如 33）
            state_steps: state 时间维步数（与 DiT 一致）
            action_steps: action 时间维步数（与 DiT 一致）
        """
        ah = self.teacher_vla.action_head
        cfg = getattr(ah, "config", None)
        if cfg is None:
            cfg = {}
        def _get(obj, key, default):
            if hasattr(obj, key):
                return getattr(obj, key)
            if isinstance(obj, dict):
                return obj.get(key, default)
            return default
        num_frames = _get(cfg, "num_frames", 21)
        # Must use model's block params (DiT uses them for num_image_blocks / action_horizon / state_horizon)
        model = getattr(ah, "model", None)
        num_frame_per_block = getattr(model, "num_frame_per_block", None) if model is not None else None
        if num_frame_per_block is None:
            num_frame_per_block = _get(cfg, "num_frame_per_block", 1)
        num_action_per_block = getattr(model, "num_action_per_block", 32) if model is not None else 32
        num_state_per_block = getattr(model, "num_state_per_block", 1) if model is not None else 1
        # Same formula as DiT L902-904: num_image_blocks = (noisy_frames-1)//num_frame_per_block, then * per_block
        latent_T = 1 + (num_frames - 1) // 4
        num_blocks = latent_T - 1
        if num_blocks <= 0:
            return num_frames, 1, num_action_per_block
        state_steps = num_blocks * num_state_per_block // num_frame_per_block
        action_steps = num_blocks * (num_action_per_block // num_frame_per_block)
        return num_frames, state_steps, action_steps

    def get_teacher_input_dims_for_frames(self, T: int) -> Tuple[int, int]:
        """
        按实际帧数 T 计算 teacher 期望的 state_steps 和 action_steps（与 DiT 内部公式一致），
        避免「图像扩成 9 帧但 action 仍按 33 帧的 96 步」导致 Dream Zero assert 失败。
        """
        ah = self.teacher_vla.action_head
        cfg = getattr(ah, "config", None) or {}
        def _get(obj, key, default):
            return getattr(obj, key, default) if hasattr(obj, key) else (obj.get(key, default) if isinstance(obj, dict) else default)
        model = getattr(ah, "model", None)
        num_frame_per_block = getattr(model, "num_frame_per_block", 2) if model is not None else _get(cfg, "num_frame_per_block", 2)
        num_action_per_block = getattr(model, "num_action_per_block", 24) if model is not None else 24
        num_state_per_block = getattr(model, "num_state_per_block", 1) if model is not None else 1
        latent_T = 1 + (T - 1) // 4
        num_blocks = latent_T - 1
        if num_blocks <= 0:
            return 1, max(1, num_action_per_block)
        state_steps = num_blocks * num_state_per_block // num_frame_per_block
        action_steps = num_blocks * (num_action_per_block // num_frame_per_block)
        return max(1, state_steps), max(1, action_steps)

    def get_teacher_action_dim(self) -> int:
        """Return the action dimension expected by the teacher (from action_head / DiT config)."""
        ah = self.teacher_vla.action_head
        dim = getattr(ah, "action_dim", None)
        if dim is not None:
            return int(dim)
        model = getattr(ah, "model", None)
        if model is not None:
            dim = getattr(model, "action_dim", None)
            if dim is not None:
                return int(dim)
        cfg = getattr(ah, "config", None)
        if isinstance(cfg, dict) and "action_dim" in cfg:
            return int(cfg["action_dim"])
        return 7  # common default for LIBERO/GR1

    def cleanup(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class MoEVLAPipeline(nn.Module):
    """
    Full distillation pipeline combining:
      1. Frozen Dream Zero teacher — last-block feature extraction via hook
      2. Trainable adapter bank with dynamic routing (MoE)
      3. Trainable Cosmos Policy student with context-augmented cross-attention

    The adapter context C_Agg is injected into the student's WanModel via
    a forward hook on the student's text_embedding Sequential module.
    After the hook, the student's cross-attention context becomes:
        context = cat(C_Agg, text_embedding(crossattn_emb))
    so the transformer blocks attend to both adapter contexts and text tokens.
    """

    def __init__(
        self,
        teacher_vla: nn.Module,
        student_model: nn.Module,
        config: MoEVLAPipelineConfig,
    ):
        super().__init__()
        self.config = config

        # --- Teacher (frozen) ---
        teacher_layer_index = getattr(config, "teacher_layer_index", 40)
        self.teacher_extractor = TeacherFeatureExtractor(teacher_vla, layer_index=teacher_layer_index)

        # --- AdapterBank (trainable) ---
        self.adapter_bank = AdapterBank(
            teacher_hidden_dim=config.teacher_hidden_dim,    
            student_hidden_dim=config.student_hidden_dim,
            adapter_bottleneck_dim=config.adapter_bottleneck_dim,
            adapter_dropout=config.adapter_dropout,
            num_adapter_output_tokens=config.num_adapter_output_tokens,
            num_specialized_experts=config.num_specialized_experts,
            top_k=config.top_k,
            gating_hidden_dim=config.gating_hidden_dim,
        )

        # --- Student (trainable backbone) ---
        self.student = student_model

        # --- Hook for context injection into student ---
        # WanModel: context = text_embedding(crossattn_emb); we hook text_embedding output.
        # MinimalV1LVGDiT: context = crossattn_proj(crossattn_emb) or raw crossattn_emb; we hook crossattn_proj if present.
        self._context_to_inject: Optional[torch.Tensor] = None
        net = getattr(student_model, "net", student_model)
        self._hook_handle = None
        self._c_agg_proj: Optional[nn.Module] = None  # 2048 -> context_dim when context_dim != student_hidden_dim

        if hasattr(net, "text_embedding"):
            self._hook_handle = net.text_embedding.register_forward_hook(self._inject_context_hook)
            logger.info("MoEVLAPipeline: registered hook on student.net.text_embedding (WanModel)")
        elif hasattr(net, "crossattn_proj"):
            # MinimalV1LVGDiT with use_crossattn_projection: project C_agg to crossattn_proj output dim if needed
            context_dim = net.crossattn_proj[0].out_features
            if config.student_hidden_dim != context_dim:
                self._c_agg_proj = nn.Linear(config.student_hidden_dim, context_dim).to(
                    device=next(student_model.parameters()).device,
                    dtype=next(student_model.parameters()).dtype,
                )
            self._hook_handle = net.crossattn_proj.register_forward_hook(self._inject_context_hook_crossattn_proj)
            logger.info(
                "MoEVLAPipeline: registered hook on student.net.crossattn_proj (MinimalV1LVGDiT), context_dim=%s",
                context_dim,
            )
        else:
            # MinimalV1LVGDiT without crossattn_proj: inject by prepending to crossattn_emb in net.forward via pre_hook.
            # crossattn_emb is args[2]; we prepend proj(C_agg) and create proj lazily on first use.
            self._hook_handle = net.register_forward_pre_hook(
                self._inject_context_pre_hook_net, with_kwargs=True
            )
            logger.info(
                "MoEVLAPipeline: registered forward_pre_hook on student.net for context injection (MinimalV1LVGDiT no crossattn_proj)."
            )

    def get_teacher_input_dims(self) -> Tuple[int, int, int]:
        """Return (num_frames, state_steps, action_steps) expected by the teacher (align with Dream Zero 原始用法)."""
        return self.teacher_extractor.get_teacher_input_dims()

    def get_teacher_action_dim(self) -> int:
        """Return the action dimension expected by the teacher."""
        return self.teacher_extractor.get_teacher_action_dim()

    def _inject_context_hook(
        self,
        module: nn.Module,
        input: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward hook on student.net.text_embedding.
        Prepends the adapter-aggregated context C_Agg to the text embedding output.

        Before hook: output = [B, L_text, student_dim=2048]
        After hook:  output = [B, 2*num_tokens + L_text, student_dim=2048]
        """
        if self._context_to_inject is not None:
            # C_Agg:  [B, 2*num_adapter_output_tokens, student_dim]
            # output: [B, L_text, student_dim]
            return torch.cat(
                [self._context_to_inject, output], dim=1
            )  # [B, 2*K + L_text, student_dim]
        return output

    def _inject_context_hook_crossattn_proj(
        self,
        module: nn.Module,
        input: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward hook on student.net.crossattn_proj (MinimalV1LVGDiT).
        Prepends C_Agg (projected to context_dim if needed) to the crossattn context.
        """
        if self._context_to_inject is not None:
            c_agg = self._c_agg_proj(self._context_to_inject) if self._c_agg_proj is not None else self._context_to_inject
            return torch.cat([c_agg, output], dim=1)
        return output

    def _inject_context_pre_hook_net(
        self,
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """
        Forward pre-hook on student.net (MinimalV1LVGDiT without crossattn_proj).
        Prepends projected C_Agg to crossattn_emb (third positional arg or kwargs['crossattn_emb']).
        Note: PyTorch only passes (module, args) to pre-hooks in some versions, so kwargs is optional.
        """
        if kwargs is None:
            kwargs = {}
        if self._context_to_inject is None:
            return args, kwargs
        # Support both positional (args[2]) and keyword (kwargs['crossattn_emb']) call styles
        crossattn_emb = args[2] if len(args) > 2 else kwargs.get("crossattn_emb")
        if crossattn_emb is None or not isinstance(crossattn_emb, torch.Tensor):
            return args, kwargs
        context_dim = crossattn_emb.shape[-1]
        if self._c_agg_proj is None or self._c_agg_proj.out_features != context_dim:
            self._c_agg_proj = nn.Linear(
                self.config.student_hidden_dim,
                context_dim,
                device=crossattn_emb.device,
                dtype=crossattn_emb.dtype,
            )
            if not hasattr(self, "_c_agg_proj_net"):
                self.add_module("_c_agg_proj_net", self._c_agg_proj)
        c_agg = self._c_agg_proj(self._context_to_inject.to(device=crossattn_emb.device, dtype=crossattn_emb.dtype))
        new_emb = torch.cat([c_agg, crossattn_emb], dim=1)
        if len(args) > 2:
            new_args = list(args)
            new_args[2] = new_emb
            # Avoid duplicate values: forward will be called as (*new_args, **kwargs).
            out_kwargs = {k: v for k, v in kwargs.items() if k not in ("x_B_C_T_H_W", "timesteps_B_T", "crossattn_emb")}
            return (tuple(new_args), out_kwargs)
        kwargs = dict(kwargs)
        kwargs["crossattn_emb"] = new_emb
        return (args, kwargs)

    def training_step(
        self,
        data_batch: Dict[str, torch.Tensor],
        iteration: int,
        teacher_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Full forward pass for adapter-augmented knowledge distillation training.

        Steps:
          1. Run frozen teacher's full forward; capture last-block H_T via hook
          2. Process H_T through AdapterBank → C_Agg + routing probs
          3. Inject C_Agg into student's cross-attention via hook
          4. Run student's normal training_step (diffusion loss on actions)

        Args:
            data_batch:      Student training batch (video, text, actions …).
            iteration:       Current training iteration.
            teacher_inputs:  **Required.**  Dream-Zero-formatted dict that
                             drives the teacher's full forward pass.
                             See ``TeacherFeatureExtractor.forward`` for the
                             required keys.

        Returns:
            output_batch:  Student's output dict (for logging).
            student_loss:  Student's diffusion loss (action prediction).
            expert_probs:  [B, num_experts] routing probabilities for aux loss.
        """
        if teacher_inputs is None:
            raise ValueError(
                "teacher_inputs is required.  Provide a Dream-Zero-formatted "
                "dict (images, text, text_attention_mask, action, state, "
                "embodiment_id, has_real_action, action_mask)."
            )

        # ===== Step 1: Teacher last-block feature extraction =====
        NAN_LOG_PATH = os.environ.get("COSMOS_NAN_LOG", "/workspace/nan.txt")
        NAN_DIAG_PATH = os.environ.get("COSMOS_NAN_DIAG", "/workspace/nan_diagnostic.jsonl")
        use_teacher_fp32 = os.environ.get("COSMOS_TEACHER_FP32", "0") == "1"
        # 设为 1 时每个 step 写一条“正常”诊断样本，用于确认路径与文件可写（不要求真的出现 NaN）
        _nan_diag_dry_run = os.environ.get("COSMOS_NAN_DIAG_DRY_RUN", "0") == "1"
        # Do NOT wrap in torch.no_grad() — see TeacherFeatureExtractor docstring.
        B = teacher_inputs["images"].shape[0]
        # Dream Zero 内部用 has_real_action 与 action_loss_per_sample (B,T) 相乘；若传入的 has_real_action 是 (B,) 且他们做了 [None,:] 会得到 (1,B)，与 (B,T) 在 dim1 不匹配。把 has_real_action 扩成 (B, T) 后无论 [None,:] 还是 [:,None] 都能与 (B,T) 正确广播，从而一次传整批。
        if B > 1 and "has_real_action" in teacher_inputs:
            hr = teacher_inputs["has_real_action"]
            action = teacher_inputs.get("action")
            if action is not None and action.dim() >= 2 and hr.dim() == 1 and hr.shape[0] == B:
                T = action.shape[1]
                teacher_inputs["has_real_action"] = hr.unsqueeze(1).expand(B, T).to(hr.device)

        if use_teacher_fp32:
            with torch.amp.autocast("cuda", dtype=torch.float32):
                h_teacher = self.teacher_extractor(teacher_inputs)  # [B, L_total, 5120]
        else:
            h_teacher = self.teacher_extractor(teacher_inputs)
        h_teacher = h_teacher.to(dtype=torch.float32)  # 保证后续 adapter 收到 float32

        # COSMOS_NAN_DIAG_DRY_RUN=1 时在首个 step 写一条正常样本，用于确认诊断路径与文件可写
        if _nan_diag_dry_run and iteration == 0:
            try:
                diag = {"iteration": 0, "dry_run": True, "h_teacher_finite": bool(torch.isfinite(h_teacher).all())}
                imgs = teacher_inputs.get("images")
                if imgs is not None and imgs.dim() >= 4:
                    v = imgs.float()
                    var_t = v.var(dim=1).mean().item()
                    diag["images_shape"] = list(imgs.shape)
                    diag["num_frames"] = imgs.shape[1]
                    diag["images_temporal_variance"] = var_t
                    diag["attention_collapse_risk"] = var_t < 1e-6
                with open(NAN_DIAG_PATH, "a") as _f:
                    _f.write(json.dumps(diag, ensure_ascii=False) + "\n")
                    _f.flush()
            except Exception:
                pass

        if not torch.isfinite(h_teacher).all():
            nan_count = torch.isnan(h_teacher).sum().item()
            inf_count = torch.isinf(h_teacher).sum().item()
            try:
                with open(NAN_LOG_PATH, "a") as _f:
                    _f.write(
                        f"[h_teacher] iter={iteration} min={h_teacher.min().item()} max={h_teacher.max().item()} "
                        f"nan_count={nan_count} inf_count={inf_count}\n"
                    )
                    _f.flush()
            except Exception:
                pass
            # 记录诊断数据，用于判断是否因「重复帧/注意力崩溃」导致 NaN
            try:
                diag = {"iteration": iteration, "h_teacher_nan_count": nan_count, "h_teacher_inf_count": inf_count}
                imgs = teacher_inputs.get("images")
                if imgs is not None and imgs.dim() >= 4:
                    t = imgs.shape[1]
                    diag["images_shape"] = list(imgs.shape)
                    diag["num_frames"] = t
                    v = imgs.float()
                    # 沿时间维方差（同一位置不同帧）：接近 0 表示多帧几乎相同，易导致注意力崩溃
                    var_t = v.var(dim=1).mean().item()
                    diag["images_temporal_variance"] = var_t
                    diag["images_min_max_mean"] = [v.min().item(), v.max().item(), v.mean().item()]
                    diag["attention_collapse_risk"] = var_t < 1e-6
                for key in ("action", "state"):
                    t = teacher_inputs.get(key)
                    if t is not None:
                        diag[f"{key}_shape"] = list(t.shape)
                text = teacher_inputs.get("text")
                if text is not None:
                    diag["text_length"] = int(text.shape[1])
                with open(NAN_DIAG_PATH, "a") as _f:
                    _f.write(json.dumps(diag, ensure_ascii=False) + "\n")
                    _f.flush()
            except Exception as e:
                logger.warning("Failed to write NaN diagnostic: %s", e)
            logger.warning(
                "NaN/Inf in h_teacher after teacher_extractor (iter=%s): min=%s max=%s nan_count=%s inf_count=%s",
                iteration, h_teacher.min().item(), h_teacher.max().item(), nan_count, inf_count,
            )

        # ===== Step 2: Adapter processing with dynamic routing =====
        c_agg, expert_probs = self.adapter_bank(
            h_teacher
        )  # c_agg: [B, 2*K, 2048], expert_probs: [B, num_experts]

        if not torch.isfinite(c_agg).all():
            try:
                with open(NAN_LOG_PATH, "a") as _f:
                    _f.write(f"[c_agg] min={c_agg.min().item()} max={c_agg.max().item()} nan_count={torch.isnan(c_agg).sum().item()}\n")
                    _f.flush()
            except Exception:
                pass
            logger.warning("NaN/Inf in c_agg after adapter_bank: min=%s max=%s", c_agg.min().item(), c_agg.max().item())
        if not torch.isfinite(expert_probs).all():
            try:
                with open(NAN_LOG_PATH, "a") as _f:
                    _f.write(f"[expert_probs] min={expert_probs.min().item()} max={expert_probs.max().item()} (check teacher or router logits)\n")
                    _f.flush()
            except Exception:
                pass
            logger.warning(
                "NaN/Inf in expert_probs: min=%s max=%s (check teacher or router logits)",
                expert_probs.min().item(), expert_probs.max().item(),
            )

        # ===== Step 3: Inject C_Agg into student cross-attention =====
        self._context_to_inject = c_agg

        # ===== Step 4: Student forward pass (diffusion training) =====
        # 学生内部 attention 使用 SDPA，仅支持 fp16/bf16，整段用 float32 autocast 会触发 No available kernel。
        # use_action_fp32 时只把 loss 转为 float32 再参与 backward，保证梯度数值稳定。
        output_batch, student_loss = self.student.training_step(data_batch, iteration)
        if getattr(self.config, "use_action_fp32", False):
            student_loss = student_loss.float()

        if not torch.isfinite(student_loss):
            try:
                with open(NAN_LOG_PATH, "a") as _f:
                    _f.write(
                        f"[student_loss] value={student_loss.item()} "
                        "(likely from teacher->adapter->condition or student diffusion)\n"
                    )
                    _f.flush()
            except Exception:
                pass
            logger.warning(
                "NaN/Inf in student_loss after student.training_step: value=%s "
                "(likely from teacher->adapter->condition or student diffusion)",
                student_loss.item(),
            )

        # ===== Step 5: Clear injection context =====
        self._context_to_inject = None

        # Attach routing probs to output for loss computation
        output_batch["expert_probs"] = expert_probs
        output_batch["adapter_context_norm"] = c_agg.norm(dim=-1).mean()

        return output_batch, student_loss, expert_probs

    @torch.no_grad()
    def generate_samples(
        self,
        data_batch: Dict[str, torch.Tensor],
        teacher_inputs: Optional[Dict[str, torch.Tensor]] = None,
        **sampling_kwargs,
    ) -> torch.Tensor:
        """
        Generate action samples at inference time with adapter augmentation.

        The teacher features are extracted once, processed through the adapter
        bank, and injected into the student's cross-attention context during
        the full diffusion sampling loop.
        """
        if teacher_inputs is None:
            raise ValueError(
                "teacher_inputs is required for generate_samples."
            )

        device = next(self.parameters()).device

        def _sync_cuda():
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)

        _t_wall_start = time.perf_counter()

        # Re-enable grad so the teacher's gradient_checkpointing path
        # correctly unpacks block outputs (see TeacherFeatureExtractor docs).
        with torch.enable_grad():
            with torch.amp.autocast("cuda", dtype=torch.float32):
                _t_teacher_start = time.perf_counter()
                _sync_cuda()
                h_teacher = self.teacher_extractor(teacher_inputs)
                _sync_cuda()
                _t_teacher_end = time.perf_counter()

        _t_adapter_start = time.perf_counter()
        _sync_cuda()
        c_agg, _ = self.adapter_bank(h_teacher.float())
        _sync_cuda()
        _t_adapter_end = time.perf_counter()

        self._context_to_inject = c_agg

        _t_student_start = time.perf_counter()
        _sync_cuda()
        samples = self.student.generate_samples_from_batch(data_batch, **sampling_kwargs)
        _sync_cuda()
        _t_student_end = time.perf_counter()

        self._context_to_inject = None

        _t_wall_end = time.perf_counter()
        _teacher_time = _t_teacher_end - _t_teacher_start
        _adapter_time = _t_adapter_end - _t_adapter_start
        _student_time = _t_student_end - _t_student_start
        _component_time = _teacher_time + _adapter_time + _student_time

        # Expose per-component timing for external logging (e.g. guided eval).
        # Ratios are computed by caller since they may want different denominators.
        self._last_component_timing = {
            "teacher_time_sec": _teacher_time,
            "adapter_time_sec": _adapter_time,
            "student_time_sec": _student_time,
            "component_time_sec": _component_time,
            "total_wall_time_sec": _t_wall_end - _t_wall_start,
        }

        return samples

    def get_trainable_parameters(self):
        """Return only the parameters that should be optimized."""
        params = []
        params.extend(self.adapter_bank.parameters())
        for p in self.student.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def freeze_teacher(self):
        """Ensure teacher is fully frozen."""
        self.teacher_extractor.teacher_vla.eval()
        for p in self.teacher_extractor.parameters():
            p.requires_grad = False

    def print_param_summary(self):
        """Print trainable/total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        adapter_params = sum(p.numel() for p in self.adapter_bank.parameters())
        student_trainable = sum(
            p.numel() for p in self.student.parameters() if p.requires_grad
        )
        teacher_params = sum(p.numel() for p in self.teacher_extractor.parameters())

        logger.info(f"Pipeline parameter summary:")
        logger.info(f"  Total params:        {total:>12,}")
        logger.info(f"  Trainable params:    {trainable:>12,}")
        logger.info(f"  Teacher (frozen):    {teacher_params:>12,}")
        logger.info(f"  Adapter bank:        {adapter_params:>12,}")
        logger.info(f"  Student (trainable): {student_trainable:>12,}")

    def cleanup(self):
        """Remove all forward hooks."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        self.teacher_extractor.cleanup()
