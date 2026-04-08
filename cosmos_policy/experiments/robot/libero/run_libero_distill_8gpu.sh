#!/bin/bash
# =============================================================================
# 8-GPU Distributed Knowledge Distillation Training Script
#
# Post-Hoc Adapter-Augmented Knowledge Distillation with Dynamic Routing
#   Teacher: Dream Zero (frozen)  — multimodal feature extraction
#   Student: Cosmos Policy (trainable) — diffusion-based action prediction
#   Adapters + Router (trainable)  — MoE context injection
#
# Architecture:
#   Teacher: Image + Text -> H_T [B, 769, 5120]
#   AdapterBank: H_T -> C_Agg [B, 32, 2048] (1 generalized + TopK specialized)
#   Student: Augmented cross-attention -> Action Chunk [B, 16, 7]
#
# Loss:
#   L_total = L_edm_diffusion + alpha * L_load_balance
#
# Usage:
#   bash run_libero_distill_8gpu.sh
#   bash run_libero_distill_8gpu.sh --resume /path/to/checkpoint.pt
# =============================================================================

set -euo pipefail

source /home/lingsheng/jiangyuhua/miniconda/etc/profile.d/conda.sh
conda activate base

# ========================== User Configuration ===============================

# -- Paths (modify these to match your environment) --
COSMOS_POLICY_ROOT="/home/lingsheng/jiangyuhua/cosmos-policy"
DREAMZERO_ROOT="/home/lingsheng/jiangyuhua/dreamzero"

STUDENT_CONFIG="${COSMOS_POLICY_ROOT}/cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py"
TEACHER_CKPT_PATH="${DREAMZERO_ROOT}/checkpoints"      # Path to Dream Zero checkpoint dir

OUTPUT_DIR="${COSMOS_POLICY_ROOT}/outputs/libero_distill_8gpu"
WANDB_PROJECT="cosmos-policy-distillation"
WANDB_RUN_NAME="libero_distill_8gpu_$(date +%Y%m%d_%H%M%S)"

# -- Hardware --
NUM_GPUS=8
GPUS="0,1,2,3,4,5,6,7"
MASTER_PORT=29500

# -- Training Hyperparameters --
MAX_ITERATIONS=100000
BATCH_SIZE_PER_GPU=4                 # Effective batch = 4 * 8 = 32
GRAD_ACCUMULATION_STEPS=1            # Effective batch = 4 * 8 * 1 = 32
LEARNING_RATE=1e-4                   # Student backbone LR
ADAPTER_LR=3e-4                      # Adapter bank LR (higher for faster convergence)
WARMUP_ITERATIONS=1000
GRAD_CLIP_NORM=1.0
SEED=42

# -- Adapter / MoE Configuration --
NUM_SPECIALIZED_EXPERTS=4            # Number of specialized adapters
TOP_K=2                              # Top-K routing
NUM_ADAPTER_OUTPUT_TOKENS=16         # Tokens per adapter output
ADAPTER_BOTTLENECK_DIM=1024          # Bottleneck MLP dim in adapters
GATING_HIDDEN_DIM=512                # Router gating network hidden dim

# -- Loss Configuration --
LOAD_BALANCE_WEIGHT=0.01             # alpha for L_load_balance
ACTION_LOSS_TYPE="l1"                # "l1" or "mse"

# -- Logging / Checkpointing --
LOG_EVERY=50
SAVE_EVERY=5000

# ========================== Environment Setup ================================

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${COSMOS_POLICY_ROOT}:${DREAMZERO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TOKENIZERS_PARALLELISM=false

# Prevent NCCL timeout on large models
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

# Memory optimization for large teacher + student co-resident on GPU
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ========================== Resume Handling ==================================

RESUME_ARG=""
if [[ "${1:-}" == "--resume" ]] && [[ -n "${2:-}" ]]; then
    RESUME_ARG="--resume_from ${2}"
    echo "[INFO] Resuming from checkpoint: ${2}"
elif [[ -d "${OUTPUT_DIR}" ]]; then
    LATEST_CKPT=$(ls -t "${OUTPUT_DIR}"/checkpoint_*.pt 2>/dev/null | head -1 || true)
    if [[ -n "${LATEST_CKPT}" ]]; then
        echo "[INFO] Found existing checkpoint: ${LATEST_CKPT}"
        echo "[INFO] Pass --resume ${LATEST_CKPT} to resume, or delete it to start fresh."
    fi
fi

# ========================== Pre-flight Checks ================================

echo "============================================================"
echo "  MoE VLA Knowledge Distillation - 8 GPU Training"
echo "============================================================"
echo ""
echo "  Student config:   ${STUDENT_CONFIG}"
echo "  Teacher ckpt:     ${TEACHER_CKPT_PATH}"
echo "  Output dir:       ${OUTPUT_DIR}"
echo "  GPUs:             ${NUM_GPUS} x ${GPUS}"
echo "  Batch size:       ${BATCH_SIZE_PER_GPU}/gpu * ${NUM_GPUS} gpus * ${GRAD_ACCUMULATION_STEPS} accum = $((BATCH_SIZE_PER_GPU * NUM_GPUS * GRAD_ACCUMULATION_STEPS)) effective"
echo "  Max iterations:   ${MAX_ITERATIONS}"
echo "  LR (student):     ${LEARNING_RATE}"
echo "  LR (adapter):     ${ADAPTER_LR}"
echo "  Experts:          ${NUM_SPECIALIZED_EXPERTS} specialized, Top-${TOP_K}"
echo "  Adapter tokens:   ${NUM_ADAPTER_OUTPUT_TOKENS}"
echo "  LB loss weight:   ${LOAD_BALANCE_WEIGHT}"
echo "  ${RESUME_ARG:+Resuming from: ${RESUME_ARG}}"
echo ""
echo "============================================================"

# Verify paths exist
if [[ ! -f "${STUDENT_CONFIG}" ]]; then
    echo "[ERROR] Student config not found: ${STUDENT_CONFIG}"
    exit 1
fi
if [[ ! -d "${TEACHER_CKPT_PATH}" ]]; then
    echo "[WARNING] Teacher checkpoint directory not found: ${TEACHER_CKPT_PATH}"
    echo "          Make sure the path is correct before training starts."
fi

mkdir -p "${OUTPUT_DIR}"

# Save this script's config for reproducibility
cp "$0" "${OUTPUT_DIR}/run_script_backup.sh" 2>/dev/null || true

# ========================== Launch Training ==================================

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 8-GPU distributed training..."
echo ""

uv run torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    --nnodes=1 \
    --node_rank=0 \
    "${COSMOS_POLICY_ROOT}/train_vla_distill.py" \
    --student_config "${STUDENT_CONFIG}" \
    --teacher_path "${TEACHER_CKPT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_iterations ${MAX_ITERATIONS} \
    --batch_size ${BATCH_SIZE_PER_GPU} \
    --learning_rate ${LEARNING_RATE} \
    --adapter_lr ${ADAPTER_LR} \
    --load_balance_weight ${LOAD_BALANCE_WEIGHT} \
    --num_specialized_experts ${NUM_SPECIALIZED_EXPERTS} \
    --top_k ${TOP_K} \
    --num_adapter_output_tokens ${NUM_ADAPTER_OUTPUT_TOKENS} \
    --grad_accumulation_steps ${GRAD_ACCUMULATION_STEPS} \
    --seed ${SEED} \
    --log_every ${LOG_EVERY} \
    --save_every ${SAVE_EVERY} \
    --use_amp \
    ${RESUME_ARG} \
    2>&1 | tee "${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed successfully."
    echo "  Checkpoints saved in: ${OUTPUT_DIR}"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training exited with code ${EXIT_CODE}."
    exit ${EXIT_CODE}
fi
