#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${1:-$LOG_DIR/gptoss120b_fuseqa_sre_full_${TS}.log}"

CMD=(
  python run_gpt_oss.py
  --base-url http://127.0.0.1:8001
  --model gpt-oss-120b
  --run-type FUSEQA-SRE
  --top-k 3
  --print-samples 0
  --log-every 50
  --write-outputs
  --run-name gptoss120b_fuseqa_sre_full
)

echo "Starting background experiment..."
echo "Log file: $LOG_FILE"

nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
PID=$!

echo "$PID" > "${LOG_FILE}.pid"
echo "PID: $PID"
echo "PID file: ${LOG_FILE}.pid"
echo "Watch logs: tail -f \"$LOG_FILE\""
