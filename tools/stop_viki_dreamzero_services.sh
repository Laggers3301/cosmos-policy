#!/usr/bin/env bash
set -euo pipefail

STATE_ROOT="${STATE_ROOT:-/data0/guoyijun/world_model_stack/state}"
VIKI_PID_FILE="${STATE_ROOT}/viki_base.pid"
DREAMZERO_PID_FILE="${STATE_ROOT}/dreamzero.pid"

stop_pid_file() {
  local pid_file="$1"
  local name="$2"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(<"${pid_file}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "[INFO] Stopping ${name} pid=${pid}"
      kill "${pid}" 2>/dev/null || true
      sleep 2
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" 2>/dev/null || true
      fi
    fi
    rm -f "${pid_file}"
  else
    echo "[INFO] ${name} pid file not found: ${pid_file}"
  fi
}

stop_pid_file "${VIKI_PID_FILE}" "VIKI-R base"
stop_pid_file "${DREAMZERO_PID_FILE}" "DreamZero"
