# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""
RoboCasa dataset wrapper for MoE VLA distillation (train_vla_distill.py).

Wraps RoboCasaDataset and adds Dream Zero–formatted teacher inputs
(teacher_images, teacher_text, teacher_action, teacher_state, etc.) so the
same batch can drive both the Cosmos Policy student and the frozen Dream Zero
teacher.

Tokenizer usage is aligned with Dream Zero (groot/vla/model/dreamzero/transform):
- Text cleaning: whitespace_clean(basic_clean(text)) when clean='whitespace'.
- Tokenizer args: padding='max_length', truncation=True, add_special_tokens=True.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.robocasa_dataset import RoboCasaDataset


def _dreamzero_text_clean(text: str) -> str:
    """
    Match Dream Zero HuggingfaceTokenizer clean='whitespace':
    basic_clean (ftfy + html unescape) then whitespace_clean (normalize spaces).
    """
    import re
    try:
        import ftfy
        import html
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
    except ImportError:
        pass
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _default_tokenizer_fn(text: str, max_length: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize for Dream Zero teacher. Tries local umt5-xxl first (google/umt5-xxl), then Qwen2-0.5B.
    Uses same options as Dream Zero: add_special_tokens=True, text cleaning (whitespace)."""
    import os
    try:
        from transformers import AutoTokenizer
        text = _dreamzero_text_clean(text)
        tokenizer = None
        for candidate in [
            "/workspace/data1/umt5-xxl",
            "/data1/lingsheng/umt5-xxl",
            os.environ.get("VLA_DISTILL_TOKENIZER_PATH"),
            "google/umt5-xxl",
        ]:
            if not candidate:
                continue
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    candidate,
                    trust_remote_code=True,
                    local_files_only=os.path.isdir(candidate) if os.path.exists(candidate) else False,
                )
                break
            except Exception:
                continue
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2-0.5B",
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2-0.5B",
                    trust_remote_code=True,
                )
        out = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )
        return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)
    except Exception as e:
        raise RuntimeError(
            "RoboCasaDistillDataset requires a tokenizer for teacher text (Dream Zero uses google/umt5-xxl). "
            "Pass teacher_tokenizer_fn or set --teacher_tokenizer_path to e.g. /data1/lingsheng/umt5-xxl, "
            "or run: hf download google/umt5-xxl --local-dir /data1/lingsheng/umt5-xxl"
        ) from e


class RoboCasaDistillDataset(Dataset):
    """
    Wraps RoboCasaDataset and adds teacher_* keys for MoE VLA distillation.

    Each sample contains:
      - All keys from RoboCasaDataset (video, t5_text_embeddings, actions, ...)
      - teacher_images: [T, H, W, 3] uint8 (left primary view, current + future frame)
      - teacher_text: [L] int64 token IDs
      - teacher_text_attention_mask: [L] int64
      - teacher_action: [chunk_size, action_dim] float32 in [-1, 1]
      - teacher_state: [2, state_dim] float32 (current + future proprio)
      - teacher_embodiment_id: int64 scalar 0
      - teacher_has_real_action: bool True
      - teacher_action_mask: [chunk_size] bool True
    """

    def __init__(
        self,
        data_dir: str,
        t5_text_embeddings_path: str,
        chunk_size: int = 32,
        teacher_tokenizer_fn: Optional[Callable[[str], tuple[torch.Tensor, torch.Tensor]]] = None,
        teacher_text_max_length: int = 512,
        teacher_action_in_minus1_1: bool = True,
        **robocasa_kwargs: Any,
    ):
        """
        Args:
            data_dir: Path to RoboCasa-Cosmos-Policy success_only (or similar).
            t5_text_embeddings_path: Path to t5_embeddings.pkl for student.
            chunk_size: Action chunk size (must match student config).
            teacher_tokenizer_fn: Optional callable(text: str) -> (input_ids, attention_mask).
                If None, uses HuggingFace Qwen2 tokenizer when available.
            teacher_text_max_length: Max sequence length for teacher text.
            teacher_action_in_minus1_1: If True, clip/scale actions to [-1, 1] for Dream Zero.
            **robocasa_kwargs: Passed to RoboCasaDataset (e.g. rollout_data_dir, use_proprio).
        """
        robocasa_kwargs.setdefault("chunk_size", chunk_size)
        robocasa_kwargs["return_raw_frames_for_teacher"] = True
        self.base = RoboCasaDataset(
            data_dir=data_dir,
            t5_text_embeddings_path=t5_text_embeddings_path,
            **robocasa_kwargs,
        )
        self.chunk_size = chunk_size
        self.teacher_text_max_length = teacher_text_max_length
        self.teacher_action_in_minus1_1 = teacher_action_in_minus1_1
        if teacher_tokenizer_fn is not None:
            self._tokenize = teacher_tokenizer_fn
        else:
            self._tokenize = lambda s: _default_tokenizer_fn(s, self.teacher_text_max_length)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base[idx]

        if "teacher_raw_frames" not in sample:
            raise KeyError(
                "RoboCasaDataset must be built with return_raw_frames_for_teacher=True. "
                "RoboCasaDistillDataset sets this automatically; check base dataset config."
            )

        raw_frames = sample["teacher_raw_frames"]  # (2, H, W, 3) uint8
        command = sample["command"]
        actions = sample["actions"]  # (chunk_size, action_dim) float32
        proprio = sample["proprio"]  # (D,) or scalar
        future_proprio = sample["future_proprio"]

        # Teacher images: keep numpy (B, T, H, W, C) after collate
        teacher_images = raw_frames  # (2, H, W, 3)

        # Teacher text: tokenize instruction
        teacher_text, teacher_text_attention_mask = self._tokenize(command)
        if isinstance(teacher_text, torch.Tensor):
            teacher_text = teacher_text.long()
            teacher_text_attention_mask = teacher_text_attention_mask.long()
        else:
            teacher_text = torch.tensor(teacher_text, dtype=torch.long)
            teacher_text_attention_mask = torch.tensor(teacher_text_attention_mask, dtype=torch.long)

        # Teacher action: ensure [-1, 1] for Dream Zero
        teacher_action = np.array(actions, dtype=np.float32)
        if self.teacher_action_in_minus1_1:
            if teacher_action.max() <= 1.0 and teacher_action.min() >= 0.0:
                teacher_action = teacher_action.astype(np.float32) * 2.0 - 1.0
            else:
                teacher_action = np.clip(teacher_action, -1.0, 1.0).astype(np.float32)
        # Guard against NaN/Inf (e.g. from rescale when curr_max==curr_min or outliers)
        teacher_action = np.nan_to_num(teacher_action, nan=0.0, posinf=1.0, neginf=-1.0)
        teacher_action = np.clip(teacher_action, -1.0, 1.0).astype(np.float32)
        teacher_action = torch.from_numpy(teacher_action)
        # 已注释：原写入 /workspace/debug.txt
        # _debug_path = os.environ.get("COSMOS_DEBUG_SHAPE_FILE", "/workspace/debug.txt")
        # try:
        #     with open(_debug_path, "a") as _f: ...
        # except Exception: pass

        # Teacher state: stack current and future proprio [2, state_dim]
        if isinstance(proprio, np.ndarray):
            state_np = np.stack([np.asarray(proprio), np.asarray(future_proprio)], axis=0)
        else:
            state_np = np.stack([np.asarray(proprio).flatten(), np.asarray(future_proprio).flatten()], axis=0)
        teacher_state = torch.from_numpy(state_np.astype(np.float32))

        teacher_embodiment_id = torch.tensor(0, dtype=torch.long)
        teacher_has_real_action = torch.tensor(True)
        teacher_action_mask = torch.ones(self.chunk_size, dtype=torch.bool)
        # 已注释：原写入 /workspace/debug.txt
        # try:
        #     with open(_debug_path, "a") as _f:
        #         _f.write(f"[robocasa_distill_dataset] __getitem__: teacher_action_mask.shape=...\n")
        # except Exception: pass

        # Drop raw frames from output; we use teacher_images
        out = {k: v for k, v in sample.items() if k != "teacher_raw_frames"}

        out["teacher_images"] = teacher_images
        out["teacher_text"] = teacher_text
        out["teacher_text_attention_mask"] = teacher_text_attention_mask
        out["teacher_action"] = teacher_action
        out["teacher_state"] = teacher_state
        out["teacher_embodiment_id"] = teacher_embodiment_id
        out["teacher_has_real_action"] = teacher_has_real_action
        out["teacher_action_mask"] = teacher_action_mask

        return out
