#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-cosmos_train}"
GPU_IDS="${GPU_IDS:-6,7}"
PORT="${PORT:-5000}"
MODEL_PATH="${MODEL_PATH:-/workspace/data1/DreamZero-DROID}"
DREAMZERO_DIR="${DREAMZERO_DIR:-/workspace/data1/guoyijun/dreamzero}"
LOG_DIR="${LOG_DIR:-/data0/guoyijun/checkpoints/logs}"
ALLOW_BUSY_GPUS="${ALLOW_BUSY_GPUS:-0}"

check_gpu_idle() {
  if [[ "$ALLOW_BUSY_GPUS" == "1" ]]; then
    return 0
  fi

  local gpu_list gpu busy_report
  IFS=',' read -r -a gpu_list <<< "$GPU_IDS"
  busy_report=""

  for gpu in "${gpu_list[@]}"; do
    gpu="${gpu// /}"
    [[ -z "$gpu" ]] && continue

    local gpu_metrics compute_pids memory_used util_gpu
    gpu_metrics="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || true)"
    compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | sed '/^$/d' || true)"

    memory_used="$(printf '%s\n' "$gpu_metrics" | awk -F',' 'NR==1 {gsub(/ /, "", $2); print $2}')"
    util_gpu="$(printf '%s\n' "$gpu_metrics" | awk -F',' 'NR==1 {gsub(/ /, "", $3); print $3}')"
    memory_used="${memory_used:-0}"
    util_gpu="${util_gpu:-0}"

    if [[ -n "$compute_pids" || "$memory_used" -gt 500 || "$util_gpu" -gt 5 ]]; then
      busy_report+=$'GPU '"$gpu"$' is busy: '"${gpu_metrics:-<no metrics>}"$'\n'
      if [[ -n "$compute_pids" ]]; then
        busy_report+=$'  compute pids: '"$compute_pids"$'\n'
      fi
    fi
  done

  if [[ -n "$busy_report" ]]; then
    echo "[ERROR] Requested GPUs are not idle. Set ALLOW_BUSY_GPUS=1 to override."
    echo "[ERROR] CUDA_VISIBLE_DEVICES=$GPU_IDS"
    printf '%s' "$busy_report"
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits || true
    exit 1
  fi
}

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "[ERROR] Container not running: $CONTAINER_NAME"
  echo "[INFO] Running containers:"
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
  exit 1
fi

check_gpu_idle

IFS=',' read -r -a gpu_arr <<< "$GPU_IDS"
NPROC_PER_NODE="${#gpu_arr[@]}"
LOG_FILE="$LOG_DIR/dreamzero_teacher_port${PORT}.log"

CMD="set -euo pipefail; \
mkdir -p '$LOG_DIR'; \
cd '$DREAMZERO_DIR'; \
source /workspace/.venv/bin/activate; \
export PYTHONPATH='$DREAMZERO_DIR'\${PYTHONPATH:+:\$PYTHONPATH}; \
export WAN21_LOCAL_DIR=/workspace/data1/Wan2.1-I2V-14B-480P; \
export CUDA_VISIBLE_DEVICES='$GPU_IDS'; \
python -m torch.distributed.run --standalone --nproc_per_node='$NPROC_PER_NODE' \
  socket_test_optimized_AR.py --port '$PORT' --enable-dit-cache --model-path '$MODEL_PATH' \
  > '$LOG_FILE' 2>&1"

echo "[INFO] container=$CONTAINER_NAME"
echo "[INFO] dreamzero_dir=$DREAMZERO_DIR"
echo "[INFO] model_path=$MODEL_PATH"
echo "[INFO] CUDA_VISIBLE_DEVICES=$GPU_IDS"
echo "[INFO] nproc_per_node=$NPROC_PER_NODE"
echo "[INFO] log_file=$LOG_FILE"

docker exec -d "$CONTAINER_NAME" bash -lc "$CMD"

echo "[OK] DreamZero teacher service started in container $CONTAINER_NAME"
echo "[OK] tail logs: docker exec $CONTAINER_NAME tail -f $LOG_FILE"
echo "[OK] test client: docker exec -it $CONTAINER_NAME bash -lc 'cd $DREAMZERO_DIR && source /workspace/.venv/bin/activate && export PYTHONPATH=$DREAMZERO_DIR\${PYTHONPATH:+:\$PYTHONPATH} && python test_client_AR.py --host localhost --port $PORT'"
