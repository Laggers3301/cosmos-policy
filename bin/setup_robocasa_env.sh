#!/usr/bin/env bash
set -euo pipefail

# Unified RoboCasa dependency setup.
# This encapsulates the commands from `orders.md` (section around lines ~346-392).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer the repo-local venv (created by uv sync below).
VENV_PY="${VENV_PY:-$ROOT_DIR/.venv/bin/python}"

echo "[1/4] uv sync (robocasa group)"
uv sync --extra cu128 --group robocasa --python 3.10

if [[ ! -f "$VENV_PY" ]]; then
  echo "ERROR: venv python not found after sync: $VENV_PY" >&2
  echo "Set VENV_PY to an existing interpreter if your venv lives elsewhere." >&2
  exit 1
fi

echo "[2/4] Install robocasa-cosmos-policy (editable)"
uv pip install -e robocasa-cosmos-policy

echo "[3/4] Install RoboCasa runtime dependencies"
uv pip install \
  torch==2.7.0+cu128 \
  torchvision==0.22.0+cu128 \
  triton==3.3.0 \
  -f https://download.pytorch.org/whl/cu128/ \
  torchaudio==2.7.0 \
  pyttsx3==2.90 \
  ray[default]==2.47.1 \
  flask \
  python-socketio>=5.13.0 \
  flask_socketio \
  lmdb \
  meshcat \
  meshcat-shapes \
  rerun-sdk==0.21.0 \
  pygame \
  sshkeyboard \
  msgpack \
  msgpack-numpy \
  pyzmq \
  PyQt6 \
  pin \
  pin-pink \
  timm \
  redis \
  datasets==3.6.0 \
  evdev \
  pybullet \
  gear \
  dm_tree \
  openai \
  tianshou==0.5.1 \
  nvidia-modelopt \
  nvidia-modelopt-core \
  tensorrt \
  openpi-client==0.1.1 \
  huggingface_hub

echo "[4/4] Pin numpy + reinstall numba"
uv pip install "numpy>=1.26.0,<2" --python "$VENV_PY"
uv pip install --reinstall numba --python "$VENV_PY"

echo "RoboCasa environment setup finished."

