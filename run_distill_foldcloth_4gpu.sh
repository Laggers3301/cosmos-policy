#!/usr/bin/env bash
# ============================================================
# 4-GPU Distill: DreamZero teacher -> Cosmos student (FoldCloth)
# Usage: bash run_distill_foldcloth_4gpu.sh
# ============================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5,6,7
export PYTHONPATH=/workspace:/workspace/data1/guoyijun/dreamzero:${PYTHONPATH:-}
export ATTENTION_BACKEND=torch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SWANLAB_API_KEY='Xs7ATabJnjXVcscqYes3V'
export WAN21_LOCAL_DIR=/workspace/data1/Wan2.1-I2V-14B-480P

OUTPUT_DIR=/data0/guoyijun/checkpoints/cosmos_policy/distill/foldcloth_distill_4gpu_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTPUT_DIR"

LOG=/data0/guoyijun/checkpoints/logs/foldcloth_distill_4gpu_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$LOG")"

echo "[INFO] Output: $OUTPUT_DIR"
echo "[INFO] Log: $LOG"

# 数据目录: preprocessed_fold_cloth_3_26
DATA_DIR=/data0/guoyijun/ALOHA-Cosmos-Policy/preprocessed_fold_cloth_3_26/fold_shirt
T5_EMB=/data0/guoyijun/ALOHA-Cosmos-Policy/preprocessed/fold_shirt/t5_embeddings.pkl

echo "[INFO] Data: $DATA_DIR"
echo "[INFO] T5: $T5_EMB"

/workspace/.venv/bin/torchrun \
  --nproc_per_node=3 \
  --master_port=12460 \
  /workspace/train_vla_distill.py \
  --student_config cosmos_policy/config/config.py \
  --student_experiment cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80 \
  --teacher_path /workspace/data1/DreamZero-DROID \
  --dreamzero_path /workspace/data1/guoyijun/dreamzero \
  --output_dir "$OUTPUT_DIR" \
  --aloha_data_dir "$DATA_DIR" \
  --aloha_t5_embeddings_path "$T5_EMB" \
  --teacher_image_text_only \
  --train_part adapter \
  --max_iterations 20000 \
  --batch_size 1 \
  --grad_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --adapter_lr 3e-4 \
  --num_specialized_experts 4 \
  --load_balance_weight 0.01 \
  --log_every 50 \
  --save_every 500 \
  --num_workers 2 \
  --swanlab_project cosmos_foldcloth \
  --swanlab_experiment foldcloth_distill_4gpu_$(date +%Y%m%d_%H%M%S) \
  --swanlab_mode cloud \
  2>&1 | tee "$LOG"