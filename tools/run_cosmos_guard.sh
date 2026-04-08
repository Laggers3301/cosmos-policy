#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -d /root/data0/guoyijun ]]; then
  DATA_ROOT_DEFAULT=/root/data0/guoyijun
elif [[ -d /data0/guoyijun ]]; then
  DATA_ROOT_DEFAULT=/data0/guoyijun
else
  DATA_ROOT_DEFAULT=/root/data0/guoyijun
fi

LOG_PREFIX="${LOG_PREFIX:-[cosmos-guard]}"

log() {
  printf '%s %s %s\n' "$LOG_PREFIX" "$(date '+%F %T')" "$*"
}

RUN_NAME="${RUN_NAME:-foldcloth_cosmos_from0_fc0326_4gpu_guarded}"
EXPERIMENT="${EXPERIMENT:-cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80}"
TRAIN_PATTERN="${TRAIN_PATTERN:-cosmos_policy.scripts.train}"
CONFIG_PATH="${CONFIG_PATH:-cosmos_policy/config/config.py}"
if [[ -z "${TORCHRUN_BIN:-}" ]]; then
  if [[ -x "$PROJECT_ROOT/.venv/bin/torchrun" ]]; then
    TORCHRUN_BIN="$PROJECT_ROOT/.venv/bin/torchrun"
  elif [[ -x /workspace/.venv/bin/torchrun ]]; then
    TORCHRUN_BIN=/workspace/.venv/bin/torchrun
  else
    TORCHRUN_BIN=torchrun
  fi
fi
MASTER_PORT="${MASTER_PORT:-12459}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

BASE_DATASETS_DIR="${BASE_DATASETS_DIR:-$DATA_ROOT_DEFAULT}"
IMAGINAIRE_OUTPUT_ROOT="${IMAGINAIRE_OUTPUT_ROOT:-$BASE_DATASETS_DIR/checkpoints}"
DATA_DIR="${DATA_DIR:-$BASE_DATASETS_DIR/ALOHA-Cosmos-Policy/preprocessed_fold_cloth_3_26/fold_shirt}"
T5_TEXT_EMBEDDINGS_PATH="${T5_TEXT_EMBEDDINGS_PATH:-$BASE_DATASETS_DIR/ALOHA-Cosmos-Policy/preprocessed/fold_shirt/t5_embeddings.pkl}"

RUN_DIR="${RUN_DIR:-$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/cosmos_v2_finetune/$EXPERIMENT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/checkpoints}"
LATEST_CHECKPOINT_FILE="${LATEST_CHECKPOINT_FILE:-$CHECKPOINT_DIR/latest_checkpoint.txt}"
STATE_DIR="${STATE_DIR:-$RUN_DIR/guard_state}"
LOG_DIR="${LOG_DIR:-$IMAGINAIRE_OUTPUT_ROOT/logs}"
PID_FILE="$STATE_DIR/train.pid"
GUARD_LOG_FILE="$STATE_DIR/guard.log"
LOCK_FILE="$STATE_DIR/guard.lock"

MEM_GUARD_GB="${MEM_GUARD_GB:-460}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
GRACEFUL_SHUTDOWN_SECONDS="${GRACEFUL_SHUTDOWN_SECONDS:-45}"
RESTART_COOLDOWN_SECONDS="${RESTART_COOLDOWN_SECONDS:-20}"
FORCE_KILL_ON_TIMEOUT="${FORCE_KILL_ON_TIMEOUT:-1}"
FORCE_FRESH_START="${FORCE_FRESH_START:-0}"
MEM_GUARD_MODE="${MEM_GUARD_MODE:-auto}"
# monitor: only log when threshold reached; restart: terminate and relaunch.
MEM_GUARD_ACTION="${MEM_GUARD_ACTION:-monitor}"
# When stopping the guard process itself, do not touch training by default.
GUARD_STOP_KILLS_TRAIN="${GUARD_STOP_KILLS_TRAIN:-0}"

DATALOADER_BATCH_SIZE="${DATALOADER_BATCH_SIZE:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-False}"
DATALOADER_LAZY_VIDEO_DECOMPRESSION="${DATALOADER_LAZY_VIDEO_DECOMPRESSION:-True}"
TRAINER_MAX_ITER="${TRAINER_MAX_ITER:-15000}"
CHECKPOINT_SAVE_ITER="${CHECKPOINT_SAVE_ITER:-100}"
JOB_GROUP="${JOB_GROUP:-}"
JOB_NAME="${JOB_NAME:-}"

WANDB_DISABLED="${WANDB_DISABLED:-true}"
SWANLAB_ENABLED="${SWANLAB_ENABLED:-1}"
SWANLAB_FAILURE_TOLERANT="${SWANLAB_FAILURE_TOLERANT:-1}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-cosmos_foldcloth}"
SWANLAB_EXPERIMENT="${SWANLAB_EXPERIMENT:-$RUN_NAME}"

TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

mkdir -p "$STATE_DIR" "$LOG_DIR" "$CHECKPOINT_DIR"
touch "$GUARD_LOG_FILE"
exec > >(tee -a "$GUARD_LOG_FILE") 2>&1

# Prevent multiple guard instances from running at the same time.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "another guard instance is already running; exiting"
  exit 0
fi

export CUDA_VISIBLE_DEVICES
export BASE_DATASETS_DIR
export IMAGINAIRE_OUTPUT_ROOT
export WANDB_DISABLED
export SWANLAB_ENABLED
export SWANLAB_FAILURE_TOLERANT
export SWANLAB_MODE
export SWANLAB_PROJECT
export SWANLAB_EXPERIMENT
export TORCHINDUCTOR_COMPILE_THREADS
export OMP_NUM_THREADS
export TOKENIZERS_PARALLELISM
export MALLOC_ARENA_MAX
export HYDRA_FULL_ERROR

if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
  export SWANLAB_API_KEY
fi

train_pid() {
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(<"$PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      printf '%s\n' "$pid"
      return 0
    fi
  fi
  return 1
}

any_train_pid() {
  pgrep -fo "$TRAIN_PATTERN" 2>/dev/null || true
}

memory_usage_cgroup_gb() {
  local bytes
  if [[ -r /sys/fs/cgroup/memory.current ]]; then
    bytes="$(< /sys/fs/cgroup/memory.current)"
  elif [[ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
    bytes="$(< /sys/fs/cgroup/memory/memory.usage_in_bytes)"
  else
    return 1
  fi
  awk -v b="$bytes" 'BEGIN { printf "%.2f", b / 1024 / 1024 / 1024 }'
}

memory_usage_rss_gb() {
  # Fallback for environments where cgroup memory files are unavailable.
  ps -eo rss=,cmd= | awk -v pat="$TRAIN_PATTERN" '
    $0 ~ pat { sum += $1 }
    END { if (sum > 0) printf "%.2f", sum / 1024 / 1024 }
  '
}

memory_usage_gb() {
  local mode="$MEM_GUARD_MODE"
  if [[ "$mode" == "disabled" ]]; then
    return 1
  fi
  if [[ "$mode" == "cgroup" ]]; then
    memory_usage_cgroup_gb
    return $?
  fi
  if [[ "$mode" == "rss" ]]; then
    memory_usage_rss_gb
    return $?
  fi
  # auto: try cgroup first, then process RSS sum.
  memory_usage_cgroup_gb 2>/dev/null || memory_usage_rss_gb
}

resolve_resume_path() {
  if [[ "$FORCE_FRESH_START" == "1" ]]; then
    return 1
  fi
  if [[ ! -s "$LATEST_CHECKPOINT_FILE" ]]; then
    return 1
  fi

  local latest
  latest="$(tr -d '[:space:]' < "$LATEST_CHECKPOINT_FILE")"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  if [[ -e "$latest" ]]; then
    printf '%s\n' "$latest"
    return 0
  fi
  if [[ -e "$CHECKPOINT_DIR/$latest" ]]; then
    printf '%s\n' "$CHECKPOINT_DIR/$latest"
    return 0
  fi
  return 1
}

stop_train() {
  local reason="${1:-requested}"
  local pid
  if ! pid="$(train_pid)"; then
    log "no live training process to stop (${reason})"
    rm -f "$PID_FILE"
    return 0
  fi

  log "sending TERM to pid=${pid} (${reason})"
  kill -TERM "$pid" 2>/dev/null || true

  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if (( waited >= GRACEFUL_SHUTDOWN_SECONDS )); then
      if [[ "$FORCE_KILL_ON_TIMEOUT" == "1" ]]; then
        log "pid=${pid} did not exit in ${GRACEFUL_SHUTDOWN_SECONDS}s, sending KILL"
        kill -KILL "$pid" 2>/dev/null || true
      else
        log "pid=${pid} still alive after ${GRACEFUL_SHUTDOWN_SECONDS}s, leaving it running"
      fi
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done

  rm -f "$PID_FILE"
}

start_train() {
  if ! command -v "$TORCHRUN_BIN" >/dev/null 2>&1 && [[ ! -x "$TORCHRUN_BIN" ]]; then
    log "torchrun not found at TORCHRUN_BIN=$TORCHRUN_BIN; skipping launch"
    return 1
  fi

  local resume_path=""
  if resume_path="$(resolve_resume_path 2>/dev/null)"; then
    log "resuming from checkpoint: $resume_path"
  else
    log "no checkpoint found, starting from scratch"
    resume_path=""
  fi

  local train_log="$LOG_DIR/${RUN_NAME}_$(date '+%F_%H-%M-%S').log"

  local -a cmd=(
    "$TORCHRUN_BIN"
    "--nproc_per_node=$NPROC_PER_NODE"
    "--master_port=$MASTER_PORT"
    -m
    cosmos_policy.scripts.train
    "--config=$CONFIG_PATH"
    --
    "experiment=$EXPERIMENT"
    "dataloader_train.dataset.data_dir=$DATA_DIR"
    "dataloader_train.dataset.t5_text_embeddings_path=$T5_TEXT_EMBEDDINGS_PATH"
    "dataloader_train.dataset.lazy_video_decompression=$DATALOADER_LAZY_VIDEO_DECOMPRESSION"
    "dataloader_train.batch_size=$DATALOADER_BATCH_SIZE"
    "dataloader_train.num_workers=$DATALOADER_NUM_WORKERS"
    "dataloader_train.persistent_workers=$DATALOADER_PERSISTENT_WORKERS"
    "trainer.max_iter=$TRAINER_MAX_ITER"
    "checkpoint.save_iter=$CHECKPOINT_SAVE_ITER"
  )

  if [[ -n "$JOB_GROUP" ]]; then
    cmd+=("job.group=$JOB_GROUP")
  fi
  if [[ -n "$JOB_NAME" ]]; then
    cmd+=("job.name=$JOB_NAME")
  fi
  if [[ -n "$resume_path" ]]; then
    cmd+=("checkpoint.load_path=$resume_path" "checkpoint.load_training_state=True")
  fi

  log "launching training -> $train_log"
  nohup "${cmd[@]}" >> "$train_log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  log "spawned pid=${pid}"
}

print_config() {
  log "project_root=$PROJECT_ROOT"
  log "data_root_default=$DATA_ROOT_DEFAULT"
  log "torchrun_bin=$TORCHRUN_BIN"
  log "run_name=$RUN_NAME experiment=$EXPERIMENT"
  log "run_dir=$RUN_DIR"
  log "checkpoint_dir=$CHECKPOINT_DIR"
  log "log_dir=$LOG_DIR"
  log "cuda_visible_devices=$CUDA_VISIBLE_DEVICES nproc=$NPROC_PER_NODE master_port=$MASTER_PORT"
  log "mem_guard_gb=$MEM_GUARD_GB mem_guard_mode=$MEM_GUARD_MODE mem_guard_action=$MEM_GUARD_ACTION check_interval_s=$CHECK_INTERVAL_SECONDS graceful_shutdown_s=$GRACEFUL_SHUTDOWN_SECONDS"
}

cleanup() {
  local code=$?
  log "guard exiting with code=$code"
}
trap cleanup EXIT
trap 'if [[ "$GUARD_STOP_KILLS_TRAIN" == "1" ]]; then stop_train "guard interrupted"; else log "guard interrupted, leaving training running"; fi' INT TERM

print_config

while true; do
  if ! train_pid >/dev/null; then
    existing_pid="$(any_train_pid)"
    if [[ -n "${existing_pid:-}" ]]; then
      log "found existing training pid=${existing_pid}; skipping new launch"
    else
      if start_train; then
        sleep "$RESTART_COOLDOWN_SECONDS"
      fi
    fi
  fi

  if [[ "$MEM_GUARD_GB" -gt 0 ]]; then
    current_mem="$(memory_usage_gb 2>/dev/null || true)"
    if [[ -n "${current_mem:-}" ]]; then
      log "memory.used=${current_mem}GiB"
      if awk -v used="$current_mem" -v limit="$MEM_GUARD_GB" 'BEGIN { exit !(used >= limit) }'; then
        if [[ "$MEM_GUARD_ACTION" == "restart" ]]; then
          stop_train "memory.used=${current_mem}GiB >= ${MEM_GUARD_GB}GiB"
          sleep "$RESTART_COOLDOWN_SECONDS"
        else
          log "memory threshold reached but action=monitor, training continues"
        fi
      fi
    else
      log "memory guard unavailable in mode=$MEM_GUARD_MODE"
    fi
  fi

  sleep "$CHECK_INTERVAL_SECONDS"
done
