"""
Training script for Post-Hoc Adapter-Augmented Knowledge Distillation
with Dynamic Routing (MoE VLA Distillation).

Usage (single GPU):
  python train_vla_distill.py \
    --student_config cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py \
    --teacher_path /path/to/dreamzero/checkpoint \
    --output_dir ./outputs/vla_distill

Usage on RoboCasa dataset (your data at e.g. /data1/lingsheng/RoboCasa-Cosmos-Policy):
  python train_vla_distill.py \
    --student_config cosmos_policy/config/config.py \
    --teacher_path /path/to/dreamzero/checkpoint \
    --output_dir ./outputs/vla_distill_robocasa \
    --robocasa_data_dir /data1/lingsheng/RoboCasa-Cosmos-Policy/success_only \
    --robocasa_t5_embeddings_path /data1/lingsheng/RoboCasa-Cosmos-Policy/success_only/t5_embeddings.pkl \
    [--robocasa_rollout_data_dir /data1/lingsheng/RoboCasa-Cosmos-Policy/all_episodes]

  Ensure BASE_DATASETS_DIR is set if your student config references it.
  Use experiment=cosmos_predict2_2b_480p_robocasa_50_demos_per_task for student config.

Usage (multi-GPU with torchrun):
  torchrun --nproc_per_node=4 train_vla_distill.py \
    --student_config cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py \
    --teacher_path /path/to/dreamzero/checkpoint \
    --output_dir ./outputs/vla_distill

Architecture:
  Teacher (Dream Zero, frozen):  Image + Text -> H_T [B, 769, 5120]
  AdapterBank (trainable):       H_T -> C_Agg [B, 2K, 2048]
  Student (Cosmos Policy, trainable): Augmented cross-attn -> Action Chunk [B, chunk_size, action_dim]

Loss:
  L_total = L_edm_diffusion + alpha * L_load_balance

NaN debugging:
  If loss is NaN, set COSMOS_DEBUG_NAN=1 to log which stage first has NaN
  (h_teacher / c_agg / expert_probs / student_loss). Common causes:
  - Teacher (Dream Zero) output contains NaN → run teacher in float32 or check checkpoint
  - Adapter/router logits overflow → expert_probs NaN
  - Student condition (injected c_agg) has NaN → propagates to diffusion loss
  - Sigma underflow in EDM denoise (division by sigma) → fixed by sigma.clamp(min=1e-7)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Optional

# Pin this process to a single GPU before torch is loaded (avoids multi-process OOM on one GPU)
if "LOCAL_RANK" in os.environ:
    _local_rank = int(os.environ["LOCAL_RANK"])
    _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if _visible:
        _gpus = [x.strip() for x in _visible.split(",")]
        if _local_rank >= len(_gpus):
            raise RuntimeError(
                f"LOCAL_RANK={_local_rank} but only {len(_gpus)} GPU(s) in CUDA_VISIBLE_DEVICES={_visible!r}. "
                f"Use --nproc_per_node={len(_gpus)} or set CUDA_VISIBLE_DEVICES to at least {_local_rank + 1} GPUs."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[_local_rank]
        os.environ["_VLA_DISTILL_ONE_GPU_PER_PROCESS"] = "1"

import numpy as np
import torch
import atexit
import signal
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vla_distill")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # --- Paths ---
    student_config: str = ""
    student_experiment: str = ""  # Experiment name for student (e.g. cosmos_predict2_2b_480p_robocasa_50_demos_per_task). Required when using config.py so model.config.net is set.
    teacher_path: str = ""
    dreamzero_path: str = "/workspace/dreamzero"  # Dream Zero repo root (where 'groot' lives). Default: env DREAMZERO_PATH or /workspace/dreamzero
    teacher_config: str = ""  # Optional: path to config.json (or dir containing it) when checkpoint dir has no config.json
    teacher_fp32: bool = False  # 若 True，Teacher 以 float32 加载并前向，避免 bfloat16 下 NaN（显存更大）
    output_dir: str = "/workspace/data1/outputs/vla_distill"
    resume_from: Optional[str] = None

    # --- RoboCasa (distillation on RoboCasa-Cosmos-Policy) ---
    robocasa_data_dir: str = ""  # e.g. /data1/lingsheng/RoboCasa-Cosmos-Policy/success_only
    robocasa_t5_embeddings_path: str = ""  # e.g. .../success_only/t5_embeddings.pkl
    robocasa_rollout_data_dir: str = ""  # optional, e.g. .../all_episodes
    teacher_tokenizer_path: str = ""  # Local path to teacher tokenizer (e.g. umt5-xxl for Dream Zero, or Qwen2-0.5B). Required when offline.
    teacher_text_max_length: int = 512

    # --- ALOHA / FoldCloth (distillation on ALOHA HDF5 data) ---
    aloha_data_dir: str = ""  # e.g. /data0/guoyijun/ALOHA-Cosmos-Policy/preprocessed_fc0326_v1/fold_shirt
    aloha_t5_embeddings_path: str = ""  # e.g. .../fold_shirt/t5_embeddings.pkl
    aloha_dataset_stats_path: str = ""  # e.g. .../fold_shirt/dataset_statistics_post_norm.json

    # --- SwanLab ---
    swanlab_project: str = ""  # SwanLab project name; empty = disabled
    swanlab_experiment: str = ""  # SwanLab experiment/run name
    swanlab_mode: str = "cloud"  # cloud or local

    # --- Pipeline / Adapter ---
    teacher_hidden_dim: int = 5120
    student_hidden_dim: int = 2048
    adapter_bottleneck_dim: int = 1024
    adapter_dropout: float = 0.1
    num_adapter_output_tokens: int = 16
    num_specialized_experts: int = 4
    top_k: int = 2
    gating_hidden_dim: int = 512

    # --- Training ---
    train_part: str = "both"  # "adapter" | "student" | "both"：只训 adapter、只训 student、或一起训
    use_action_fp32: bool = False  # 若 True，student_loss 转为 float32 再 backward（学生前向仍 bf16，因 SDPA 仅支持 fp16/bf16）
    max_iterations: int = 100_000
    batch_size: int = 4
    learning_rate: float = 1e-4
    adapter_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_iterations: int = 1000
    grad_clip_norm: float = 1.0
    grad_accumulation_steps: int = 1
    use_amp: bool = True

    # --- Loss ---
    load_balance_weight: float = 0.01
    action_loss_type: str = "l1"

    # --- Logging / Checkpointing ---
    log_every: int = 50
    save_every: int = 100
    validate_every: int = 2000
    # 相对 output_dir 的 loss 曲线 CSV；设为空字符串则关闭写入
    loss_curve_csv: str = "loss_curve.csv"
    num_workers: int = 4
    seed: int = 42

    # --- Teacher input mode (for distillation without action/state space mismatch) ---
    teacher_image_text_only: bool = False  # If True, feed teacher only images+text; use empty action/state so DiT runs image+text only and we distill from that representation.
    teacher_layer_index: int = 40  # 取 Teacher 第几层输出作为 H_T（1-based）。设为 14 则只保留前 14 层，显存更小。


# ---------------------------------------------------------------------------
# Helper: Parameter group construction
# ---------------------------------------------------------------------------
def build_optimizer(pipeline: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """
    Build optimizer. train_part 决定只优化 adapter、只优化 student、或两者：
      - adapter: 仅 AdapterBank 参数，lr=adapter_lr
      - student: 仅学生参数，lr=learning_rate
      - both: 两组参数，各自学习率
    """
    adapter_params = []
    student_params = []

    for name, param in pipeline.named_parameters():
        if not param.requires_grad:
            continue
        if "adapter_bank" in name:
            adapter_params.append(param)
        else:
            student_params.append(param)

    param_groups = []
    if adapter_params:
        param_groups.append({
            "params": adapter_params,
            "lr": config.adapter_lr,
            "weight_decay": config.weight_decay,
            "name": "adapter_bank",
        })
    if student_params:
        param_groups.append({
            "params": student_params,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "name": "student_backbone",
        })

    if not param_groups:
        raise ValueError(
            "No trainable parameters. Check train_part: adapter/student/both and that the selected part has requires_grad=True."
        )

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    logger.info(
        f"Optimizer: AdamW | train_part={config.train_part} | "
        f"Adapter params: {sum(p.numel() for p in adapter_params):,} @ lr={config.adapter_lr} | "
        f"Student params: {sum(p.numel() for p in student_params):,} @ lr={config.learning_rate}"
    )
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """Cosine annealing with linear warmup."""
    import math as _math

    def lr_lambda(step: int) -> float:
        if step < config.warmup_iterations:
            return step / max(config.warmup_iterations, 1)
        progress = (step - config.warmup_iterations) / max(
            config.max_iterations - config.warmup_iterations, 1
        )
        return 0.5 * (1.0 + _math.cos(_math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Model Loading Utilities
# ---------------------------------------------------------------------------
def load_teacher_model(teacher_path: str, device: torch.device, dreamzero_path: str = "", teacher_config: str = "", teacher_fp32: bool = False) -> nn.Module:
    """
    Load the Dream Zero teacher VLA model.

    Expects the checkpoint directory to contain model weights; config can be in that dir
    or provided via teacher_config (path to config.json or dir containing it).
    """
    # Force this process to use only its assigned GPU before any Dream Zero / CUDA code runs
    torch.cuda.set_device(device.index if device.index is not None else 0)
    repo_root = dreamzero_path or os.environ.get("DREAMZERO_PATH", "/workspace/dreamzero")
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from groot.vla.model.dreamzero.base_vla import VLA

    logger.info(f"Loading teacher model from: {teacher_path}")
    teacher = VLA.from_pretrained(teacher_path, config_path=teacher_config or None)
    teacher_dtype = torch.float32 if teacher_fp32 else torch.bfloat16
    teacher = teacher.to(device=device, dtype=teacher_dtype)
    if teacher_fp32:
        logger.info("Teacher loaded in float32 (teacher_fp32=True) for numerical stability.")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info("Teacher model loaded and frozen.")
    return teacher


def load_student_model(student_config_path: str, device: torch.device, student_experiment: str = "") -> nn.Module:
    """
    Load the Cosmos Policy student diffusion model.
    Uses the Cosmos Policy config system (LazyConfig) to instantiate the model.
    """
    torch.cuda.set_device(device.index if device.index is not None else 0)
    from cosmos_policy._src.imaginaire.lazy_config import instantiate
    from cosmos_policy._src.imaginaire.config import load_config

    logger.info(f"Loading student model from config: {student_config_path}")
    opts = ["--", "model.config.fsdp_shard_size=1"]  # Always disable FSDP for distill (single-GPU or per-process single GPU).
    if student_experiment:
        opts.extend([f"experiment={student_experiment}"])
    config = load_config(student_config_path, opts)
    # Force FSDP off so single-GPU run (no RANK) and multi-GPU with CUDA_VISIBLE_DEVICES pinning both work.
    try:
        if hasattr(config, "model") and hasattr(config.model, "config"):
            config.model.config.fsdp_shard_size = 1
            # Force flash_attention_2 if available (installed via pip)
            config.model.config._attn_implementation = "flash_attention_2"
    except Exception:
        pass
    student = instantiate(config.model)
    student = student.to(device=device)
    logger.info("Student model loaded.")
    return student


# ---------------------------------------------------------------------------
# Checkpoint Utilities
# ---------------------------------------------------------------------------
def save_checkpoint(
    pipeline: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    iteration: int,
    config: TrainConfig,
    output_dir: str,
):
    """Save trainable pipeline state (adapter bank + student) and optimizer."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, f"checkpoint_{iteration:08d}.pt")

    state = {
        "iteration": iteration,
        "adapter_bank": pipeline.adapter_bank.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": asdict(config),
    }
    # Save student trainable params only (not teacher)
    student_trainable = {
        k: v
        for k, v in pipeline.student.state_dict().items()
        if any(
            p.data_ptr() == v.data_ptr()
            for p in pipeline.student.parameters()
            if p.requires_grad
        )
    }
    state["student_trainable"] = student_trainable

    torch.save(state, ckpt_path)
    logger.info(f"Checkpoint saved: {ckpt_path}")
    
    # Update latest_checkpoint.txt
    latest_path = os.path.join(output_dir, "latest_checkpoint.txt")
    with open(latest_path, "w") as f:
        f.write(ckpt_path)


def load_checkpoint(
    pipeline: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    ckpt_path: str,
) -> int:
    """Load checkpoint and return the iteration to resume from."""
    logger.info(f"Resuming from checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")

    pipeline.adapter_bank.load_state_dict(state["adapter_bank"])
    if "student_trainable" in state:
        pipeline.student.load_state_dict(state["student_trainable"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])

    iteration = state["iteration"]
    logger.info(f"Resumed from iteration {iteration}")
    return iteration


def _is_cuda_oom_error(err: BaseException) -> bool:
    msg = str(err)
    return "CUDA out of memory" in msg or "cuda out of memory" in msg


def _dump_oom_artifacts(config: TrainConfig, err: BaseException) -> None:
    """
    Dump GPU/process memory snapshot when CUDA OOM happens.
    This is best-effort and should never mask the original exception.
    """
    try:
        local_rank = os.environ.get("LOCAL_RANK", "unknown")
        pid = os.getpid()
        ts = time.strftime("%Y%m%d-%H%M%S")
        dump_root = os.path.join(config.output_dir, "oom_dumps")
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"oom--rank{local_rank}--pid{pid}--{ts}.json")

        def _run_cmd(cmd: list[str]) -> str:
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                return out
            except Exception as e:  # pragma: no cover
                return f"<failed to run {cmd}: {e}>"

        env_snapshot = {
            "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "RANK": os.environ.get("RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        }

        # GPU/process snapshot
        nvidia_smi_free = _run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ]
        )
        nvidia_smi_apps = _run_cmd(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_bus_id,used_memory", "--format=csv,noheader"]
        )

        # Only query ps for PIDs that nvidia-smi reports, to keep output small.
        pids: set[str] = set()
        for line in nvidia_smi_apps.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts and parts[0].isdigit():
                pids.add(parts[0])

        ps_cmd = ["ps", "-o", "pid=,ppid=,rss=,cmd=", "-p", ",".join(sorted(pids))] if pids else ["ps", "-o", "pid=,ppid=,rss=,cmd="]
        ps_snapshot = _run_cmd(ps_cmd)

        # Per-process CUDA memory summary (best-effort)
        cuda_summary = ""
        try:
            if torch.cuda.is_available():
                dev_idx = torch.cuda.current_device()
                cuda_summary = torch.cuda.memory_summary(device=torch.device(f"cuda:{dev_idx}"))
        except Exception as e:  # pragma: no cover
            cuda_summary = f"<failed to get torch.cuda.memory_summary: {e}>"

        payload = {
            "error_type": type(err).__name__,
            "error_message": str(err),
            "traceback": traceback.format_exc(),
            "env": env_snapshot,
            "nvidia_smi_gpu_mem": nvidia_smi_free,
            "nvidia_smi_processes": nvidia_smi_apps,
            "ps_snapshot": ps_snapshot,
            "torch_cuda_memory_summary": cuda_summary,
        }

        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        # Never hide the original OOM.
        return


def _scalar_to_float(x) -> float:
    """Convert torch.Tensor scalar (or Python number) to float."""
    if hasattr(x, "item") and callable(getattr(x, "item")):
        return float(x.item())
    return float(x)


def append_loss_curve_row(
    output_dir: str,
    csv_basename: str,
    row: dict,
) -> None:
    """Append one training metrics row to CSV under output_dir (rank0 only)."""
    if not csv_basename:
        return
    path = os.path.join(output_dir, csv_basename)
    fieldnames = [
        "iteration",
        "epoch",
        "avg_total_loss",
        "avg_student_edm_loss",
        "avg_load_balance_loss",
        "last_step_student_edm_loss",
        "lr_adapter_bank",
        "lr_student_backbone",
        "it_per_sec",
        "demo_sample_action_l1_loss",
        "demo_sample_action_mse_loss",
    ]
    os.makedirs(output_dir, exist_ok=True)
    write_header = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(config: TrainConfig):
    """
    Main training loop for MoE VLA knowledge distillation.

    Training flow per iteration:
      1. Teacher (frozen): extract H_T [B, 769, 5120] from Image + Text
      2. AdapterBank: H_T -> C_Agg [B, 2K, 2048] + routing probabilities
      3. Student: inject C_Agg into cross-attention, run diffusion training_step
      4. Loss: L_edm + alpha * L_load_balance
      5. Backward + optimizer step
    """
    # --- Distributed: set device first; init process group only after models are loaded ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    one_gpu_per_process = os.environ.get("_VLA_DISTILL_ONE_GPU_PER_PROCESS") == "1"
    if is_distributed and not one_gpu_per_process:
        logger.warning(
            "Multi-GPU run without CUDA_VISIBLE_DEVICES pinning: each process may use the wrong GPU and cause OOM. "
            "Start with: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_vla_distill.py ..."
        )
    if is_distributed or os.environ.get("LOCAL_RANK") is not None:
        torch.cuda.set_device(local_rank if not one_gpu_per_process else 0)

    device = torch.device("cuda:0" if one_gpu_per_process else f"cuda:{local_rank}")
    is_main = local_rank == 0

    if is_main:
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"Training config:\n{json.dumps(asdict(config), indent=2)}")

    # --- Seed ---
    torch.manual_seed(config.seed + local_rank)
    torch.cuda.manual_seed(config.seed + local_rank)

    # --- Load models (each rank loads on its own GPU before any collective) ---
    # Teacher 前向精度：1=float32（稳定但显存大），0=bf16（省显存，可配合 NaN 诊断确认是否注意力崩溃）
    os.environ["COSMOS_TEACHER_FP32"] = "1" if config.teacher_fp32 else "0"
    teacher = load_teacher_model(config.teacher_path, device, config.dreamzero_path, config.teacher_config, config.teacher_fp32)
    student = load_student_model(config.student_config, device, config.student_experiment)

    # --- Now init process group (NCCL) for DDP and dataloader ---
    if is_distributed:
        pg = dist.init_process_group(backend="nccl"); _set_dist_process_group(pg)

    # --- Build pipeline ---
    from cosmos_policy.pipeline.moe_vla_pipeline import MoEVLAPipeline, MoEVLAPipelineConfig
    from cosmos_policy.losses.distillation_losses import DistillationLoss, DistillationLossConfig

    pipeline_config = MoEVLAPipelineConfig(
        teacher_hidden_dim=config.teacher_hidden_dim,
        student_hidden_dim=config.student_hidden_dim,
        adapter_bottleneck_dim=config.adapter_bottleneck_dim,
        adapter_dropout=config.adapter_dropout,
        num_adapter_output_tokens=config.num_adapter_output_tokens,
        num_specialized_experts=config.num_specialized_experts,
        top_k=config.top_k,
        gating_hidden_dim=config.gating_hidden_dim,
        use_action_fp32=getattr(config, "use_action_fp32", False),
        teacher_layer_index=getattr(config, "teacher_layer_index", 40),
    )

    pipeline = MoEVLAPipeline(
        teacher_vla=teacher,
        student_model=student,
        config=pipeline_config,
    )
    pipeline.freeze_teacher()
    pipeline.to(device)

    # 按 train_part 冻结其中一部分，实现「只训 adapter」或「只训 student」
    train_part = (getattr(config, "train_part", "both") or "both").strip().lower()
    if train_part not in ("adapter", "student", "both"):
        train_part = "both"
    if train_part == "adapter":
        for p in pipeline.adapter_bank.parameters():
            p.requires_grad = True
        for p in pipeline.student.parameters():
            p.requires_grad = False
        if is_main:
            logger.info("Train part: adapter only (student frozen).")
    elif train_part == "student":
        for p in pipeline.adapter_bank.parameters():
            p.requires_grad = False
        for p in pipeline.student.parameters():
            p.requires_grad = True
        if is_main:
            logger.info("Train part: student only (adapter frozen).")
    else:
        if is_main:
            logger.info("Train part: both (adapter + student).")

    if is_main:
        pipeline.print_param_summary()

    # --- DDP wrapper (only wraps trainable parts) ---
    if is_distributed:
        pipeline = nn.parallel.DistributedDataParallel(
            pipeline,
            device_ids=[0 if os.environ.get("_VLA_DISTILL_ONE_GPU_PER_PROCESS") == "1" else local_rank],
            find_unused_parameters=True,
        )
        pipeline_module = pipeline.module
    else:
        pipeline_module = pipeline

    # --- Loss ---
    loss_config = DistillationLossConfig(
        load_balance_weight=config.load_balance_weight,
        action_loss_type=config.action_loss_type,
        num_experts=config.num_specialized_experts,
    )
    distillation_loss_fn = DistillationLoss(loss_config).to(device)

    # --- Optimizer + Scheduler ---
    optimizer = build_optimizer(pipeline_module, config)
    scheduler = build_scheduler(optimizer, config)

    # --- AMP scaler ---
    scaler = GradScaler(enabled=config.use_amp)

    # --- Resume ---
    start_iteration = 0
    if config.resume_from:
        start_iteration = load_checkpoint(
            pipeline_module, optimizer, scheduler, config.resume_from
        )

    # --- Data loader ---
    if config.aloha_data_dir and config.aloha_t5_embeddings_path:
        dataloader_train = _build_aloha_distill_dataloader(config, is_distributed, world_size, local_rank)
        if is_main:
            logger.info("Using ALOHA distillation dataset: %s", config.aloha_data_dir)
    elif config.robocasa_data_dir and config.robocasa_t5_embeddings_path:
        dataloader_train = _build_robocasa_distill_dataloader(config, is_distributed, world_size, local_rank)
        if is_main:
            logger.info("Using RoboCasa distillation dataset: %s", config.robocasa_data_dir)
    else:
        logger.info(
            "NOTE: Set --aloha_data_dir / --robocasa_data_dir to train on real data."
        )
        dataloader_train = _build_placeholder_dataloader(config, is_distributed, world_size, local_rank)

    # --- SwanLab init ---
    swanlab_run = None
    if is_main and config.swanlab_project:
        try:
            import swanlab
            swanlab_run = swanlab.init(
                project=config.swanlab_project,
                experiment_name=config.swanlab_experiment or None,
                mode=config.swanlab_mode,
                config=asdict(config),
            )
            logger.info("SwanLab enabled: project=%s experiment=%s", config.swanlab_project, config.swanlab_experiment)
        except Exception as e:
            logger.warning("SwanLab init failed: %s", e)
            swanlab_run = None

    # --- Training loop ---
    pipeline.train()
    pipeline_module.freeze_teacher()

    running_loss = 0.0
    running_lb_loss = 0.0
    running_edm_loss = 0.0
    t_start = time.time()
    iteration = start_iteration
    epoch = 0

    while iteration < config.max_iterations:
        if is_distributed and hasattr(dataloader_train.sampler, "set_epoch"):
            dataloader_train.sampler.set_epoch(epoch)

        for data_batch in dataloader_train:
            if iteration >= config.max_iterations:
                break

            # Move batch to GPU (convert numpy to tensor if needed)
            def _to_device(v):
                if isinstance(v, torch.Tensor):
                    return v.to(device, non_blocking=True)
                if isinstance(v, np.ndarray):
                    return torch.from_numpy(v).to(device, non_blocking=True)
                return v

            data_batch = {k: _to_device(v) for k, v in data_batch.items()}

            # Build Dream-Zero-formatted teacher inputs from the data batch.
            # Keys prefixed with "teacher_" are extracted and stripped of the prefix.
            teacher_inputs = {
                k[len("teacher_"):]: v
                for k, v in data_batch.items()
                if k.startswith("teacher_")
            }
            # Pass flag for image+text-only teacher (no action/state from dataset)
            teacher_inputs["_teacher_image_text_only"] = config.teacher_image_text_only
            if not teacher_inputs:
                raise ValueError(
                    "data_batch contains no 'teacher_*' keys.  The dataset "
                    "must provide Dream-Zero-formatted fields prefixed with "
                    "'teacher_' (e.g. teacher_images, teacher_text, …)."
                )
            # 已注释：原写入 /workspace/debug.txt 的 batch shape 调试
            # _debug_path = os.environ.get("COSMOS_DEBUG_SHAPE_FILE", "/workspace/debug.txt")
            # try:
            #     with open(_debug_path, "a") as _f:
            #         a = teacher_inputs.get("action")
            #         m = teacher_inputs.get("action_mask")
            #         _f.write(...)
            # except Exception: pass

            # --- Forward pass ---
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=config.use_amp):
                output_batch, student_loss, expert_probs = pipeline_module.training_step(
                    data_batch, iteration, teacher_inputs=teacher_inputs
                )

                total_loss, loss_dict = distillation_loss_fn(
                    student_loss=student_loss,
                    expert_probs=expert_probs,
                )

                # Scale for gradient accumulation
                scaled_loss = total_loss / config.grad_accumulation_steps

            # --- Debug: print loss every step (set COSMOS_DEBUG_LOSS_EVERY_STEP=1 to enable) ---
            if os.environ.get("COSMOS_DEBUG_LOSS_EVERY_STEP") == "1" and is_main:
                logger.info(
                    f"[DEBUG iter {iteration+1}] loss={total_loss.item():.4f} "
                    f"edm={loss_dict['student_edm_loss']:.4f} lb={loss_dict['load_balance_loss']:.4f}"
                )

            # --- 将有助于检查 NaN 来源的数据写入 /workspace/nan.txt ---
            NAN_LOG_PATH = "/workspace/nan.txt"
            if is_main:
                try:
                    edm = loss_dict["student_edm_loss"]
                    lb = loss_dict["load_balance_loss"]
                    total_f = torch.isfinite(total_loss)
                    edm_f = torch.isfinite(edm)
                    lb_f = torch.isfinite(lb)
                    if not (total_f and edm_f and lb_f) or os.environ.get("COSMOS_DEBUG_NAN") == "1":
                        with open(NAN_LOG_PATH, "a") as _f:
                            total_s = "nan" if not total_f else f"{total_loss.item():.6f}"
                            edm_s = "nan" if not edm_f else f"{edm.item():.6f}"
                            lb_s = "nan" if not lb_f else f"{lb.item():.6f}"
                            _f.write(
                                f"iter={iteration+1} total_loss={total_s} student_edm_loss={edm_s} load_balance_loss={lb_s}\n"
                            )
                            _f.flush()
                except Exception:
                    pass

            # --- Backward pass ---
            scaler.scale(scaled_loss).backward()

            # --- Optimizer step (after accumulation) ---
            if (iteration + 1) % config.grad_accumulation_steps == 0:
                # 第一次 step 前打一条 log，便于确认前向/反向已跑完（若随后 OOM 可看到 loss）
                if is_main and (iteration + 1) == config.grad_accumulation_steps:
                    logger.info(
                        "[First optimizer step] grad_accumulation done, loss=%.4f (forward/backward completed, stepping optimizer next).",
                        total_loss.item(),
                    )
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in pipeline.parameters() if p.requires_grad],
                    config.grad_clip_norm,
                )
                # 缓解 optimizer.step 时显存碎片导致的 OOM：先同步再清缓存，再 step（务必设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True）
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # --- Logging ---
            running_loss += total_loss.item()
            running_lb_loss += loss_dict["load_balance_loss"].item()
            running_edm_loss += _scalar_to_float(loss_dict["student_edm_loss"])

            # 每 500 iter 清理一次 CPU cache 防止 OOM
            if is_main and (iteration + 1) % 500 == 0:
                import gc
                import psutil
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                mem_used_gb = psutil.Process().memory_info().rss / 1024**3
                logger.info(f"[Cache cleanup] iter={iteration+1} CPU_mem={mem_used_gb:.1f}GB")

            if is_main and (iteration + 1) % config.log_every == 0:
                avg_loss = running_loss / config.log_every
                avg_lb = running_lb_loss / config.log_every
                avg_edm = running_edm_loss / config.log_every
                elapsed = time.time() - t_start
                it_per_sec = config.log_every / elapsed

                lr_by_name = {g.get("name", ""): g["lr"] for g in optimizer.param_groups}
                lr_adapter = lr_by_name.get("adapter_bank", 0.0)
                lr_student = lr_by_name.get("student_backbone", 0.0)

                edm_last = loss_dict["student_edm_loss"]
                edm_last_f = _scalar_to_float(edm_last)

                logger.info(
                    f"[Iter {iteration+1:>7d}/{config.max_iterations}] "
                    f"loss={avg_loss:.4f} "
                    f"edm={avg_edm:.4f} "
                    f"lb={avg_lb:.4f} "
                    f"lr_adpt={lr_adapter:.2e} "
                    f"lr_stud={lr_student:.2e} "
                    f"it/s={it_per_sec:.2f}"
                )

                if "demo_sample_action_l1_loss" in output_batch:
                    logger.info(
                        f"  action_l1={output_batch['demo_sample_action_l1_loss']:.4f} "
                        f"action_mse={output_batch['demo_sample_action_mse_loss']:.4f}"
                    )

                curve_row = {
                    "iteration": iteration + 1,
                    "epoch": epoch,
                    "avg_total_loss": avg_loss,
                    "avg_student_edm_loss": avg_edm,
                    "avg_load_balance_loss": avg_lb,
                    "last_step_student_edm_loss": edm_last_f,
                    "lr_adapter_bank": lr_adapter,
                    "lr_student_backbone": lr_student,
                    "it_per_sec": it_per_sec,
                }
                if "demo_sample_action_l1_loss" in output_batch:
                    curve_row["demo_sample_action_l1_loss"] = _scalar_to_float(
                        output_batch["demo_sample_action_l1_loss"]
                    )
                if "demo_sample_action_mse_loss" in output_batch:
                    curve_row["demo_sample_action_mse_loss"] = _scalar_to_float(
                        output_batch["demo_sample_action_mse_loss"]
                    )
                try:
                    append_loss_curve_row(
                        config.output_dir,
                        getattr(config, "loss_curve_csv", "loss_curve.csv"),
                        curve_row,
                    )
                except Exception as e:
                    logger.warning("Failed to append loss curve CSV: %s", e)

                if swanlab_run is not None:
                    try:
                        import swanlab
                        swanlab.log(curve_row, step=iteration + 1)
                    except Exception as e:
                        logger.warning("SwanLab log failed: %s", e)

                running_loss = 0.0
                running_lb_loss = 0.0
                running_edm_loss = 0.0
                t_start = time.time()

            # --- Checkpointing ---
            if is_main and (iteration + 1) % config.save_every == 0:
                save_checkpoint(
                    pipeline_module,
                    optimizer,
                    scheduler,
                    iteration + 1,
                    config,
                    config.output_dir,
                )

            iteration += 1

        epoch += 1

    # --- Final save ---
    if is_main:
        save_checkpoint(
            pipeline_module,
            optimizer,
            scheduler,
            iteration,
            config,
            config.output_dir,
        )
        logger.info("Training complete.")

    if swanlab_run is not None:
        try:
            swanlab_run.finish()
        except Exception:
            pass

    pipeline_module.cleanup()

    if is_distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# RoboCasa distillation data loader
# ---------------------------------------------------------------------------
def _load_teacher_tokenizer(path: str, max_length: int):
    """Load tokenizer from a local path (e.g. Dream Zero checkpoint dir). Returns a callable or None.
    Callable matches Dream Zero: text cleaning (whitespace) + add_special_tokens=True."""
    import os
    try:
        from transformers import AutoTokenizer
        from cosmos_policy.datasets.robocasa_distill_dataset import _dreamzero_text_clean
    except ImportError:
        return None
    if not path or not os.path.isdir(path):
        return None
    try:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    except Exception:
        return None

    def _fn(s: str):
        s = _dreamzero_text_clean(s)
        out = tok(
            s,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )
        return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)

    return _fn


def _build_robocasa_distill_dataloader(
    config: TrainConfig,
    is_distributed: bool,
    world_size: int,
    local_rank: int,
) -> DataLoader:
    """Build DataLoader over RoboCasa-Cosmos-Policy with teacher_* keys for distillation."""
    from cosmos_policy.datasets.robocasa_distill_dataset import RoboCasaDistillDataset

    teacher_tokenizer_fn = None
    if config.teacher_tokenizer_path:
        from transformers import AutoTokenizer
        from cosmos_policy.datasets.robocasa_distill_dataset import _dreamzero_text_clean
        _tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        _max_len = config.teacher_text_max_length

        def _fn(s: str):
            s = _dreamzero_text_clean(s)
            out = _tokenizer(
                s,
                return_tensors="pt",
                padding="max_length",
                max_length=_max_len,
                truncation=True,
                add_special_tokens=True,
            )
            return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)

        teacher_tokenizer_fn = _fn
        logger.info("Using teacher tokenizer from: %s", config.teacher_tokenizer_path)
    elif config.teacher_path:
        teacher_tokenizer_fn = _load_teacher_tokenizer(config.teacher_path, config.teacher_text_max_length)
        if teacher_tokenizer_fn is not None:
            logger.info("Using teacher tokenizer from Dream Zero checkpoint: %s", config.teacher_path)
    if teacher_tokenizer_fn is None:
        # Dream Zero 使用 UMT5-XXL，优先从本地 umt5-xxl 加载；其次 env；再 Qwen fallback
        import os
        for candidate in [
            os.environ.get("VLA_DISTILL_TOKENIZER_PATH"),
            "/workspace/data1/umt5-xxl",
            "/data1/lingsheng/umt5-xxl",
            os.path.join(os.path.dirname(config.teacher_path or ""), "umt5-xxl"),
            "/workspace/data1/Qwen2-0.5B",
            os.path.join(os.path.dirname(config.teacher_path or ""), "Qwen2-0.5B"),
        ]:
            if not candidate or not os.path.isdir(candidate):
                continue
            teacher_tokenizer_fn = _load_teacher_tokenizer(candidate, config.teacher_text_max_length)
            if teacher_tokenizer_fn is not None:
                logger.info("Using teacher tokenizer from local path: %s", candidate)
                break
    if teacher_tokenizer_fn is None and (not is_distributed or local_rank == 0):
        logger.warning(
            "No local tokenizer found (checked teacher_path, VLA_DISTILL_TOKENIZER_PATH, /workspace/data1/umt5-xxl, Qwen2-0.5B). "
            "For Dream Zero use: hf download google/umt5-xxl --local-dir /data1/lingsheng/umt5-xxl then set --teacher_tokenizer_path or mount to /workspace/data1/umt5-xxl.",
        )

    dataset = RoboCasaDistillDataset(
        data_dir=config.robocasa_data_dir,
        t5_text_embeddings_path=config.robocasa_t5_embeddings_path,
        chunk_size=32,
        teacher_tokenizer_fn=teacher_tokenizer_fn,
        teacher_text_max_length=config.teacher_text_max_length,
        use_image_aug=True,
        use_stronger_image_aug=True,
        use_wrist_images=True,
        use_third_person_images=True,
        use_proprio=True,
        normalize_proprio=True,
        normalize_actions=True,
        num_duplicates_per_image=4,
        rollout_data_dir=config.robocasa_rollout_data_dir or "",
        demonstration_sampling_prob=0.5,
        success_rollout_sampling_prob=0.5,
        return_value_function_returns=True,
        gamma=0.99,
    )

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        if is_distributed
        else None
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# ALOHA distillation data loader
# ---------------------------------------------------------------------------
def _build_aloha_distill_dataloader(
    config: TrainConfig,
    is_distributed: bool,
    world_size: int,
    local_rank: int,
) -> DataLoader:
    from cosmos_policy.datasets.aloha_distill_dataset import AlohaDistillDataset

    dataset = AlohaDistillDataset(
        data_dir=config.aloha_data_dir,
        t5_text_embeddings_path=config.aloha_t5_embeddings_path,
        dataset_stats_path=config.aloha_dataset_stats_path,
        is_train=True,
        chunk_size=50,
    )

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        if is_distributed
        else None
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )


# Placeholder data loader
# ---------------------------------------------------------------------------
def _build_placeholder_dataloader(
    config: TrainConfig,
    is_distributed: bool,
    world_size: int,
    local_rank: int,
) -> DataLoader:
    """
    Creates a minimal placeholder dataloader that produces dummy tensors
    matching the Cosmos Policy data format.

    REPLACE THIS with your actual dataset (e.g., LIBERODataset, RoboCasaDataset).

    Expected data_batch keys for the full pipeline:

      Student (Cosmos Policy) keys:
        video:                         [B, C=3, T, H=224, W=224]  or uint8 [B, T, C, H, W]
        t5_text_embeddings:            [B, L_text, 4096]
        t5_text_mask:                  [B, L_text]
        actions:                       [B, chunk_size, action_dim]
        action_latent_idx:             [B]  (int)
        proprio:                       [B, proprio_dim]
        current_proprio_latent_idx:    [B]  (int)
        future_proprio:                [B, proprio_dim]
        future_proprio_latent_idx:     [B]  (int)
        future_image_latent_idx:       [B]  (int)
        future_wrist_image_latent_idx: [B]  (int)
        rollout_data_mask:             [B]  (0=demo, 1=rollout)
        world_model_sample_mask:       [B]  (0 or 1)
        value_function_sample_mask:    [B]  (0 or 1)
        value_function_return:         [B]
        value_latent_idx:              [B]  (int)

      Teacher (Dream Zero) keys — prefixed with "teacher_":
        teacher_images:              [B, T, H, W, C]  video frames
        teacher_text:                [B, L]  token IDs
        teacher_text_attention_mask: [B, L]  attention mask
        teacher_action:              [B, T_a, action_dim]  in [-1, 1]
        teacher_state:               [B, T_s, state_dim]
        teacher_embodiment_id:       [B]
        teacher_has_real_action:     bool or [B]
        teacher_action_mask:         [B, T_a]
    """
    chunk_size = 16
    action_dim = 7
    proprio_dim = 9
    # [MODIFIED 2026-03-23] Reason: smoke test placeholder must match student expected video length (state_t->33 frames)
    num_frames = 33
    text_len = 512
    H, W = 224, 224
    state_t = 9

    class _PlaceholderDataset(torch.utils.data.Dataset):
        def __init__(self, length: int = 1000):
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            # Dream Zero teacher action/state dimensions (match LIBERO config)
            teacher_action_len = 16
            teacher_state_dim = 7

            return {
                # ----- Student (Cosmos Policy) fields -----
                # [MODIFIED 2026-03-23] Reason:
                # student text2world_model requires uint8 video input before normalization.
                "video": torch.randint(0, 256, (3, num_frames, H, W), dtype=torch.uint8),
                # [ORIGINAL CODE - kept for traceability]
                # "t5_text_embeddings": torch.randn(text_len, 4096),
                # [MODIFIED 2026-03-23] Reason: student cross-attn expects text context dim=1024 in current config.
                "t5_text_embeddings": torch.randn(text_len, 1024),
                # [MODIFIED 2026-03-23] Reason: keep dtype consistent with real datasets
                "t5_text_mask": torch.ones(text_len, dtype=torch.int64),
                # [MODIFIED 2026-03-23] Reason: required by video2world conditioner
                "fps": 16,
                "padding_mask": torch.zeros(1, H, W),
                "image_size": H * torch.ones(4),
                "actions": torch.randn(chunk_size, action_dim).clamp(-1, 1),
                "action_latent_idx": torch.tensor(state_t - 2, dtype=torch.long),
                "proprio": torch.randn(proprio_dim),
                "current_proprio_latent_idx": torch.tensor(-1, dtype=torch.long),
                "future_proprio": torch.randn(proprio_dim),
                "future_proprio_latent_idx": torch.tensor(-1, dtype=torch.long),
                "future_image_latent_idx": torch.tensor(-1, dtype=torch.long),
                "future_image2_latent_idx": torch.tensor(-1, dtype=torch.long),
                "future_wrist_image_latent_idx": torch.tensor(-1, dtype=torch.long),
                "rollout_data_mask": torch.tensor(0, dtype=torch.long),
                "world_model_sample_mask": torch.tensor(0, dtype=torch.long),
                "value_function_sample_mask": torch.tensor(0, dtype=torch.long),
                "value_function_return": torch.tensor(0.0),
                "value_latent_idx": torch.tensor(state_t - 1, dtype=torch.long),
                # ----- Teacher (Dream Zero) fields (prefixed "teacher_") -----
                # REPLACE with real data from your dataset.
                # [MODIFIED 2026-03-23] Reason:
                # keep teacher image input in uint8 as well, consistent with DreamZero preprocessing path.
                "teacher_images": torch.randint(0, 256, (num_frames, H, W, 3), dtype=torch.uint8),
                "teacher_text": torch.randint(0, 32000, (text_len,)),
                "teacher_text_attention_mask": torch.ones(text_len, dtype=torch.long),
                "teacher_action": torch.randn(teacher_action_len, action_dim).clamp(-1, 1),
                "teacher_state": torch.randn(num_frames - 1, teacher_state_dim),
                "teacher_embodiment_id": torch.tensor(0, dtype=torch.long),
                "teacher_has_real_action": torch.tensor(True),
                "teacher_action_mask": torch.ones(teacher_action_len, dtype=torch.bool),
            }

    dataset = _PlaceholderDataset()

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
        if is_distributed
        else None
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="MoE VLA Knowledge Distillation Training")
    parser.add_argument("--student_config", type=str, required=True, help="Path to Cosmos Policy config")
    parser.add_argument("--student_experiment", type=str, default="", help="Experiment name so model.config.net is set (e.g. cosmos_predict2_2b_480p_robocasa_50_demos_per_task). Required when using config.py.")
    parser.add_argument("--teacher_path", type=str, required=True, help="Path to Dream Zero checkpoint")
    parser.add_argument("--dreamzero_path", type=str, default="", help="Dream Zero repo root (where 'groot' lives). Default: env DREAMZERO_PATH or /workspace/dreamzero")
    parser.add_argument("--teacher_config", type=str, default="", help="Path to config.json (or dir containing it) when checkpoint dir has no config.json")
    parser.add_argument("--output_dir", type=str, default="./outputs/vla_distill")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_iterations", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adapter_lr", type=float, default=3e-4)
    parser.add_argument("--load_balance_weight", type=float, default=0.01)
    parser.add_argument("--num_specialized_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--num_adapter_output_tokens", type=int, default=16)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument(
        "--loss_curve_csv",
        type=str,
        default="loss_curve.csv",
        help="Basename of CSV under --output_dir to append loss curves each --log_every steps; empty string disables.",
    )
    parser.add_argument("--robocasa_data_dir", type=str, default="", help="RoboCasa success_only dir (e.g. /data1/lingsheng/RoboCasa-Cosmos-Policy/success_only)")
    parser.add_argument("--robocasa_t5_embeddings_path", type=str, default="", help="Path to t5_embeddings.pkl for RoboCasa")
    parser.add_argument("--robocasa_rollout_data_dir", type=str, default="", help="Optional: RoboCasa all_episodes dir")
    parser.add_argument("--aloha_data_dir", type=str, default="", help="ALOHA HDF5 data dir (e.g. /data0/guoyijun/ALOHA-Cosmos-Policy/preprocessed_fc0326_v1/fold_shirt)")
    parser.add_argument("--aloha_t5_embeddings_path", type=str, default="", help="Path to t5_embeddings.pkl for ALOHA")
    parser.add_argument("--aloha_dataset_stats_path", type=str, default="", help="Path to dataset_statistics_post_norm.json for ALOHA")
    parser.add_argument("--swanlab_project", type=str, default="", help="SwanLab project name; empty disables SwanLab")
    parser.add_argument("--swanlab_experiment", type=str, default="", help="SwanLab experiment/run name")
    parser.add_argument("--swanlab_mode", type=str, default="cloud", help="SwanLab mode: cloud or local")
    parser.add_argument("--teacher_tokenizer_path", type=str, default="", help="Local path to teacher tokenizer (e.g. umt5-xxl for Dream Zero). Use: hf download google/umt5-xxl --local-dir /data1/lingsheng/umt5-xxl")
    parser.add_argument("--teacher_text_max_length", type=int, default=512, help="Max token length for teacher text")
    parser.add_argument("--teacher_image_text_only", action="store_true", help="Feed teacher only images+text (empty action/state). Use when distilling to avoid action/state space mismatch across datasets.")
    parser.add_argument("--teacher_fp32", action="store_true", help="Load and run Teacher in float32 to avoid NaN in bfloat16 (uses more GPU memory).")
    parser.add_argument("--train_part", type=str, default="both", choices=["adapter", "student", "both"], help="Train adapter only, student only, or both. Default: both.")
    parser.add_argument("--use_action_fp32", action="store_true", help="Run student action forward and loss in float32; rest of training stays bf16.")
    parser.add_argument("--teacher_layer_index", type=int, default=40, help="Use teacher output from this layer (1-based). Default 40 (last). Use 14 to keep only first 14 layers and save GPU memory.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader worker processes.")

    args = parser.parse_args()

    config = TrainConfig(
        student_config=args.student_config,
        student_experiment=args.student_experiment,
        teacher_path=args.teacher_path,
        dreamzero_path=args.dreamzero_path,
        teacher_config=args.teacher_config,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        adapter_lr=args.adapter_lr,
        load_balance_weight=args.load_balance_weight,
        num_specialized_experts=args.num_specialized_experts,
        top_k=args.top_k,
        num_adapter_output_tokens=args.num_adapter_output_tokens,
        use_amp=args.use_amp,
        grad_accumulation_steps=args.grad_accumulation_steps,
        seed=args.seed,
        log_every=args.log_every,
        save_every=args.save_every,
        loss_curve_csv=args.loss_curve_csv,
        robocasa_data_dir=args.robocasa_data_dir,
        robocasa_t5_embeddings_path=args.robocasa_t5_embeddings_path,
        robocasa_rollout_data_dir=args.robocasa_rollout_data_dir,
        aloha_data_dir=args.aloha_data_dir,
        aloha_t5_embeddings_path=args.aloha_t5_embeddings_path,
        aloha_dataset_stats_path=args.aloha_dataset_stats_path,
        swanlab_project=args.swanlab_project,
        swanlab_experiment=args.swanlab_experiment,
        swanlab_mode=args.swanlab_mode,
        teacher_tokenizer_path=args.teacher_tokenizer_path,
        teacher_text_max_length=args.teacher_text_max_length,
        teacher_image_text_only=args.teacher_image_text_only,
        teacher_fp32=args.teacher_fp32,
        train_part=args.train_part,
        use_action_fp32=args.use_action_fp32,
        teacher_layer_index=args.teacher_layer_index,
        num_workers=args.num_workers,
    )
    return config


# Global variable to track NCCL process group
_dist_process_group = None

def _set_dist_process_group(pg):
    global _dist_process_group
    _dist_process_group = pg

def _cleanup_nccl():
    import torch.distributed as dist
    if _dist_process_group is not None:
        try:
            dist.destroy_process_group()
        except:
            pass

# Register cleanup for normal exit and signals
atexit.register(_cleanup_nccl)

def _signal_handler(signum, frame):
    _cleanup_nccl()
    import sys
    sys.exit(128 + signum)

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

if __name__ == "__main__":
    config = parse_args()
    try:
        train(config)
    except BaseException as e:
        if _is_cuda_oom_error(e):
            _dump_oom_artifacts(config, e)
        raise
