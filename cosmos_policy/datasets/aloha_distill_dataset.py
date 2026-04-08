# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""
ALOHA HDF5 dataset wrapper for MoE VLA distillation (train_vla_distill.py).

Wraps AlohaDataset and adds Dream Zero-formatted teacher inputs
(teacher_images, teacher_text, teacher_action, teacher_state, etc.).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_policy.datasets.aloha_dataset import ALOHADataset


AlohaDataset = ALOHADataset  # alias for readability


def _dreamzero_text_clean(text: str) -> str:
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


def _default_tokenizer_fn(text: str, max_length: int = 512):
    try:
        from transformers import AutoTokenizer
        for candidate in [
            "/root/data1/lingsheng/umt5-xxl",
            "/workspace/data1/umt5-xxl",
        ]:
            import os
            if os.path.isdir(candidate):
                tok = AutoTokenizer.from_pretrained(candidate)
                out = tok(
                    text, return_tensors="pt", padding="max_length",
                    max_length=max_length, truncation=True, add_special_tokens=True,
                )
                return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)
        # Qwen fallback
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        out = tok(
            text, return_tensors="pt", padding="max_length",
            max_length=max_length, truncation=True, add_special_tokens=True,
        )
        return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)
    except Exception:
        ids = torch.zeros(max_length, dtype=torch.long)
        mask = torch.zeros(max_length, dtype=torch.long)
        return ids, mask


class AlohaDistillDataset(Dataset):
    """
    Wraps AlohaDataset to produce both student (Cosmos Policy) and teacher (Dream Zero)
    fields in each batch sample.

    Teacher fields added:
        teacher_images              [T_t, H, W, C]  uint8
        teacher_text                [L]  long
        teacher_text_attention_mask [L]  long
        teacher_action              [chunk_size, action_dim]  float32 in [-1,1]
        teacher_state               [2, state_dim]  float32
        teacher_embodiment_id       scalar long
        teacher_has_real_action     bool tensor
        teacher_action_mask         [chunk_size] bool
    """

    def __init__(
        self,
        data_dir: str,
        t5_text_embeddings_path: str,
        is_train: bool = True,
        chunk_size: int = 50,
        num_teacher_frames: int = 6,
        teacher_image_size: int = 224,
        teacher_action_in_minus1_1: bool = True,
        teacher_image_text_only: bool = False,
        tokenizer_fn: Optional[Callable] = None,
        dataset_stats_path: str = "",  # kept for API compat, not passed to ALOHADataset (it auto-loads from data_dir)
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        return_value_function_returns: bool = True,
    ):
        self.chunk_size = chunk_size
        self.num_teacher_frames = num_teacher_frames
        self.teacher_image_size = teacher_image_size
        self.teacher_action_in_minus1_1 = teacher_action_in_minus1_1
        self.teacher_image_text_only = teacher_image_text_only
        self.tokenizer_fn = tokenizer_fn or _default_tokenizer_fn

        self._base = AlohaDataset(
            data_dir=data_dir,
            is_train=is_train,
            chunk_size=chunk_size,
            t5_text_embeddings_path=t5_text_embeddings_path,
            use_proprio=True,
            normalize_actions=normalize_actions,
            normalize_proprio=normalize_proprio,
            lazy_video_decompression=True,
            num_duplicates_per_image=num_duplicates_per_image,
            return_value_function_returns=return_value_function_returns,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        for _retry in range(len(self._base)):
            try:
                sample = self._base[(idx + _retry) % len(self._base)]
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("All samples in dataset are corrupt")

        # --- Teacher images: subsample from student video ---
        # student video shape: [C, T, H, W] (from AlohaDataset)
        video = sample["video"]  # [C, T, H, W] float tensor
        C, T, H, W = video.shape
        # Pick num_teacher_frames evenly spaced
        indices = np.linspace(0, T - 1, self.num_teacher_frames, dtype=int)
        frames = video[:, indices, :, :]  # [C, T_t, H, W]
        # Convert to [T_t, H, W, C] uint8
        frames = frames.permute(1, 2, 3, 0)  # [T_t, H, W, C]
        frames = (frames * 255).clamp(0, 255).to(torch.uint8)
        teacher_images = frames

        # --- Teacher text ---
        command = sample["command"]
        command_clean = _dreamzero_text_clean(command)
        if self.teacher_image_text_only:
            teacher_text = torch.zeros(512, dtype=torch.long)
            teacher_text_attention_mask = torch.zeros(512, dtype=torch.long)
        else:
            teacher_text, teacher_text_attention_mask = self.tokenizer_fn(command_clean)
            if not isinstance(teacher_text, torch.Tensor):
                teacher_text = torch.tensor(teacher_text, dtype=torch.long)
            if not isinstance(teacher_text_attention_mask, torch.Tensor):
                teacher_text_attention_mask = torch.tensor(teacher_text_attention_mask, dtype=torch.long)

        # --- Teacher action ---
        actions = sample["actions"]  # [chunk_size, action_dim] tensor or ndarray
        if isinstance(actions, torch.Tensor):
            teacher_action = actions.numpy().astype(np.float32)
        else:
            teacher_action = np.array(actions, dtype=np.float32)

        if self.teacher_action_in_minus1_1:
            if teacher_action.max() <= 1.0 and teacher_action.min() >= 0.0:
                teacher_action = teacher_action * 2.0 - 1.0
            else:
                teacher_action = np.clip(teacher_action, -1.0, 1.0)
        teacher_action = np.nan_to_num(teacher_action, nan=0.0, posinf=1.0, neginf=-1.0)
        teacher_action = np.clip(teacher_action, -1.0, 1.0).astype(np.float32)
        teacher_action = torch.from_numpy(teacher_action)

        # Pad or trim to chunk_size
        T_a = teacher_action.shape[0]
        if T_a < self.chunk_size:
            pad = torch.zeros(self.chunk_size - T_a, teacher_action.shape[1], dtype=torch.float32)
            teacher_action = torch.cat([teacher_action, pad], dim=0)
        else:
            teacher_action = teacher_action[:self.chunk_size]

        # --- Teacher state: stack current + future proprio ---
        proprio = sample["proprio"]
        future_proprio = sample["future_proprio"]
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.numpy()
        if isinstance(future_proprio, torch.Tensor):
            future_proprio = future_proprio.numpy()
        state_np = np.stack([
            np.asarray(proprio).flatten(),
            np.asarray(future_proprio).flatten(),
        ], axis=0).astype(np.float32)
        teacher_state = torch.from_numpy(state_np)

        teacher_embodiment_id = torch.tensor(0, dtype=torch.long)
        teacher_has_real_action = torch.tensor(True)
        teacher_action_mask = torch.ones(self.chunk_size, dtype=torch.bool)

        out = dict(sample)
        out["teacher_images"] = teacher_images
        out["teacher_text"] = teacher_text
        out["teacher_text_attention_mask"] = teacher_text_attention_mask
        out["teacher_action"] = teacher_action
        out["teacher_state"] = teacher_state
        out["teacher_embodiment_id"] = teacher_embodiment_id
        out["teacher_has_real_action"] = teacher_has_real_action
        out["teacher_action_mask"] = teacher_action_mask
        return out
