#!/usr/bin/env bash
# ============================================================
# 直接训练 Cosmos (不是蒸馏) - FoldCloth ALOHA 版本
# Usage: bash run_cosmos_foldcloth_4gpu.sh
# ============================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES=5,6,7
export BASE_DATASETS_DIR=/data0/guoyijun
export WANDB_DISABLED=true
export IMAGINAIRE_OUTPUT_ROOT=/data0/guoyijun/checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SWANLAB_ENABLED=1
export SWANLAB_API_KEY='Xs7ATabJnjXVcscqYes3V'
export SWANLAB_PROJECT='cosmos_foldcloth'
export SWANLAB_MODE='cloud'
export SWANLAB_FAILURE_TOLERANT=0

# 创建输出目录
mkdir -p /data0/guoyijun/checkpoints/logs

# 数据目录
DATA_DIR=/data0/guoyijun/ALOHA-Cosmos-Policy/preprocessed_fold_cloth_3_26/fold_shirt
T5_EMB=/data0/guoyijun/ALOHA-Cosmos-Policy/preprocessed/fold_shirt/t5_embeddings.pkl

# 实验名: 使用 ALOHA 版本
EXPERIMENT="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80"

# 输出配置
RUN_NAME="foldcloth_cosmos_aloha_4gpu_$(date +%Y%m%d_%H%M%S)"
LOG=/data0/guoyijun/checkpoints/logs/${RUN_NAME}.log

echo "=========================================="
echo "[INFO] Experiment: $EXPERIMENT"
echo "[INFO] Data: $DATA_DIR"
echo "[INFO] Log: $LOG"
echo "=========================================="

/workspace/.venv/bin/torchrun \
  --nproc_per_node=3 \
  --master_port=12341 \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="$EXPERIMENT" \
  dataloader_train.dataset.data_dir="$DATA_DIR" \
  dataloader_train.dataset.t5_text_embeddings_path="$T5_EMB" \
  dataloader_train.dataset.lazy_video_decompression=True \
  dataloader_train.batch_size=1 \
  dataloader_train.num_workers=2 \
  dataloader_train.persistent_workers=True \
  job.wandb_mode=disabled \
  trainer.max_iter=20000 \
  checkpoint.save_iter=500 \
  job.group="guoyijun_foldcloth_cosmos" \
  job.name="$RUN_NAME" \
  2>&1 | tee "$LOG"