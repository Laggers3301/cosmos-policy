#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="${SESSION_NAME:-cosmos_guard}"
GUARD_SCRIPT="$PROJECT_ROOT/tools/run_cosmos_guard.sh"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but was not found."
  exit 1
fi

if [[ ! -x "$GUARD_SCRIPT" ]]; then
  chmod +x "$GUARD_SCRIPT"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists."
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" "cd '$PROJECT_ROOT' && bash '$GUARD_SCRIPT'"
echo "Started tmux session '$SESSION_NAME'."
echo "Attach with: tmux attach -t $SESSION_NAME"