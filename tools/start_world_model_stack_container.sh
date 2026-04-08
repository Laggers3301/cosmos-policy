#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-world_model_stack}"
IMAGE_NAME="${IMAGE_NAME:-openpi:installed}"

WORKSPACE_HOST="${WORKSPACE_HOST:-/data1/lingsheng/guoyijun/cosmos-policy}"
DREAMZERO_HOST="${DREAMZERO_HOST:-/data1/lingsheng/guoyijun/dreamzero}"

HOST_VIKI_PORT="${HOST_VIKI_PORT:-9000}"
HOST_DREAMZERO_PORT="${HOST_DREAMZERO_PORT:-9003}"

VIKI_GPU_IDS="${VIKI_GPU_IDS:-2}"
DREAMZERO_GPU_IDS="${DREAMZERO_GPU_IDS:-0,1}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  -p "${HOST_VIKI_PORT}:9000" \
  -p "${HOST_DREAMZERO_PORT}:9003" \
  --entrypoint /bin/bash \
  -v /root/.cache:/root/.cache \
  -v /data0:/data0 \
  -v /data1:/workspace/data1 \
  -v "${WORKSPACE_HOST}:/workspace" \
  -v "${DREAMZERO_HOST}:/workspace/dreamzero" \
  -w /workspace \
  "${IMAGE_NAME}" -lc 'sleep infinity'

docker exec -d \
  -e VIKI_GPU_IDS="${VIKI_GPU_IDS}" \
  -e DREAMZERO_GPU_IDS="${DREAMZERO_GPU_IDS}" \
  "${CONTAINER_NAME}" \
  bash -lc '/workspace/tools/start_viki_dreamzero_services.sh'

echo "[INFO] Container started: ${CONTAINER_NAME}"
echo "[INFO] Host endpoints"
echo "  VIKI-R base: http://127.0.0.1:${HOST_VIKI_PORT}"
echo "  DreamZero:   ws://127.0.0.1:${HOST_DREAMZERO_PORT}"
echo "  DreamZero health: http://127.0.0.1:${HOST_DREAMZERO_PORT}/healthz"
echo "[INFO] Enter container"
echo "  docker exec -it ${CONTAINER_NAME} bash"
echo "[INFO] Stop services"
echo "  docker exec ${CONTAINER_NAME} bash -lc /workspace/tools/stop_viki_dreamzero_services.sh"
echo "[INFO] Stop container"
echo "  docker rm -f ${CONTAINER_NAME}"
