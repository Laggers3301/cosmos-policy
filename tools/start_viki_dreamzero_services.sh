#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
DREAMZERO_CODE_DIR="${DREAMZERO_CODE_DIR:-${WORKSPACE_ROOT}/dreamzero}"
VIKI_ROOT="${VIKI_ROOT:-${WORKSPACE_ROOT}/data1/wxr/VIKI-R}"

DREAMZERO_PY="${DREAMZERO_PY:-/data0/conda_env/dreamzero/bin/python}"
VIKI_VLLM_BIN="${VIKI_VLLM_BIN:-/data0/conda_env/roboviki/bin/vllm}"

DREAMZERO_CKPT="${DREAMZERO_CKPT:-${WORKSPACE_ROOT}/data1/lingsheng/DreamZero-DROID}"
if [[ -z "${VIKI_MODEL_DIR:-}" ]]; then
  if [[ -d "${VIKI_ROOT}/saves/qwen2.5_vl-3b/full/fold_cloth_wxr_sft" ]]; then
    VIKI_MODEL_DIR="${VIKI_ROOT}/saves/qwen2.5_vl-3b/full/fold_cloth_wxr_sft"
  else
    VIKI_MODEL_DIR="${VIKI_ROOT}/saves/qwen2.5_vl-3b/full/viki_1_sft"
  fi
fi

LOG_ROOT="${LOG_ROOT:-/data0/guoyijun/world_model_stack/logs}"
STATE_ROOT="${STATE_ROOT:-/data0/guoyijun/world_model_stack/state}"

VIKI_GPU_IDS="${VIKI_GPU_IDS:-2}"
VIKI_HOST="${VIKI_HOST:-0.0.0.0}"
VIKI_PORT="${VIKI_PORT:-9000}"
VIKI_MODEL_NAME="${VIKI_MODEL_NAME:-viki-base}"
VIKI_MAX_MODEL_LEN="${VIKI_MAX_MODEL_LEN:-4096}"

DREAMZERO_GPU_IDS="${DREAMZERO_GPU_IDS:-0,1}"
DREAMZERO_PORT="${DREAMZERO_PORT:-9003}"
DREAMZERO_ENABLE_DIT_CACHE="${DREAMZERO_ENABLE_DIT_CACHE:-1}"

mkdir -p "${LOG_ROOT}" "${STATE_ROOT}"

VIKI_PID_FILE="${STATE_ROOT}/viki_base.pid"
DREAMZERO_PID_FILE="${STATE_ROOT}/dreamzero.pid"

count_csv_items() {
  local raw="$1"
  awk -F',' '{print NF}' <<<"${raw// /}"
}

stop_pid_file() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(<"${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      sleep 2
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" 2>/dev/null || true
      fi
    fi
    rm -f "${pid_file}"
  fi
}

require_path() {
  local target="$1"
  local desc="$2"
  if [[ ! -e "${target}" ]]; then
    echo "[ERROR] Missing ${desc}: ${target}" >&2
    exit 1
  fi
}

require_path "${DREAMZERO_CODE_DIR}" "DreamZero code dir"
require_path "${VIKI_ROOT}" "VIKI-R root"
require_path "${DREAMZERO_PY}" "DreamZero python"
require_path "${VIKI_VLLM_BIN}" "VIKI-R vllm binary"
require_path "${DREAMZERO_CKPT}" "DreamZero checkpoint"
require_path "${VIKI_MODEL_DIR}" "VIKI-R model dir"

stop_pid_file "${VIKI_PID_FILE}"
stop_pid_file "${DREAMZERO_PID_FILE}"

echo "[INFO] Starting VIKI-R base service"
echo "[INFO]   model=${VIKI_MODEL_DIR}"
echo "[INFO]   gpus=${VIKI_GPU_IDS}"
echo "[INFO]   port=${VIKI_PORT}"

nohup env \
  CUDA_VISIBLE_DEVICES="${VIKI_GPU_IDS}" \
  TOKENIZERS_PARALLELISM=false \
  "${VIKI_VLLM_BIN}" serve "${VIKI_MODEL_DIR}" \
    --served-model-name "${VIKI_MODEL_NAME}" \
    --host "${VIKI_HOST}" \
    --port "${VIKI_PORT}" \
    --trust-remote-code \
    --max-model-len "${VIKI_MAX_MODEL_LEN}" \
  > "${LOG_ROOT}/viki_base.log" 2>&1 &
echo $! > "${VIKI_PID_FILE}"

DREAMZERO_NPROC="$(count_csv_items "${DREAMZERO_GPU_IDS}")"
echo "[INFO] Starting DreamZero service"
echo "[INFO]   ckpt=${DREAMZERO_CKPT}"
echo "[INFO]   gpus=${DREAMZERO_GPU_IDS}"
echo "[INFO]   port=${DREAMZERO_PORT}"

pushd "${DREAMZERO_CODE_DIR}" >/dev/null
if [[ "${DREAMZERO_ENABLE_DIT_CACHE}" == "1" ]]; then
  DREAMZERO_CACHE_FLAG="--enable-dit-cache"
else
  DREAMZERO_CACHE_FLAG=""
fi

nohup env \
  CUDA_VISIBLE_DEVICES="${DREAMZERO_GPU_IDS}" \
  WAN21_LOCAL_DIR="${WAN21_LOCAL_DIR:-${WORKSPACE_ROOT}/data1/Wan2.1-I2V-14B-480P}" \
  PYTHONPATH="${DREAMZERO_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${DREAMZERO_PY}" -m torch.distributed.run --standalone --nproc_per_node="${DREAMZERO_NPROC}" \
    socket_test_optimized_AR.py \
    --port "${DREAMZERO_PORT}" \
    ${DREAMZERO_CACHE_FLAG} \
    --model-path "${DREAMZERO_CKPT}" \
  > "${LOG_ROOT}/dreamzero.log" 2>&1 &
echo $! > "${DREAMZERO_PID_FILE}"
popd >/dev/null

sleep 5

echo "[INFO] PIDs"
echo "  VIKI-R: $(<"${VIKI_PID_FILE}")"
echo "  DreamZero: $(<"${DREAMZERO_PID_FILE}")"
echo "[INFO] Logs"
echo "  ${LOG_ROOT}/viki_base.log"
echo "  ${LOG_ROOT}/dreamzero.log"
echo "[INFO] Endpoints"
echo "  VIKI-R base: http://127.0.0.1:${VIKI_PORT}"
echo "  DreamZero:   ws://127.0.0.1:${DREAMZERO_PORT}"
echo "  DreamZero health: http://127.0.0.1:${DREAMZERO_PORT}/healthz"
