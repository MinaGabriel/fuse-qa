#!/usr/bin/env bash
set -euo pipefail

# Safe launcher for a single vLLM instance on port 8001.
# - Refuses to start if port 8001 is already in use (default).
# - Optional forced reclaim: FORCE_RECLAIM_PORT=1 ./start_vllm_8001.sh
# - Writes PID + log files under ./logs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
MODEL_PATH="${MODEL_PATH:-/data/LLMs/Llama-3.1-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-llama-3.1-8b-instruct}"
DTYPE="${DTYPE:-bfloat16}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-120}"
FORCE_RECLAIM_PORT="${FORCE_RECLAIM_PORT:-0}"

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${LOG_FILE:-$LOG_DIR/vllm_${SERVED_MODEL_NAME}_${PORT}_${TS}.log}"
PID_FILE="${PID_FILE:-$LOG_DIR/vllm_${PORT}.pid}"

print_port_holders() {
  ss -ltnp | awk -v p=":${PORT}" '$4 ~ p {print}'
}

port_in_use() {
  ss -ltnp | awk -v p=":${PORT}" '$4 ~ p {found=1} END{exit found?0:1}'
}

extract_port_pids() {
  ss -ltnp | awk -v p=":${PORT}" '
    $4 ~ p {
      line=$0
      while (match(line, /pid=[0-9]+/)) {
        print substr(line, RSTART+4, RLENGTH-4)
        line=substr(line, RSTART+RLENGTH)
      }
    }
  ' | sort -u
}

if port_in_use; then
  echo "Port ${PORT} is already in use:"
  print_port_holders

  if [[ "$FORCE_RECLAIM_PORT" == "1" ]]; then
    echo "FORCE_RECLAIM_PORT=1 set: reclaiming port ${PORT}..."
    PIDS="$(extract_port_pids || true)"
    if [[ -n "${PIDS}" ]]; then
      # Continue stopped jobs first, then terminate.
      echo "$PIDS" | xargs -r kill -CONT || true
      sleep 0.3
      echo "$PIDS" | xargs -r kill -TERM || true
      sleep 1
      STILL="$(extract_port_pids || true)"
      if [[ -n "${STILL}" ]]; then
        echo "$STILL" | xargs -r kill -KILL || true
      fi
      sleep 0.5
    fi
  else
    echo "Refusing to start another vLLM on port ${PORT}."
    echo "If you really want to reclaim this port, run:"
    echo "  FORCE_RECLAIM_PORT=1 $0"
    exit 1
  fi
fi

if port_in_use; then
  echo "Port ${PORT} is still occupied after reclaim attempt:"
  print_port_holders
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "PID file exists and process is alive: PID=${OLD_PID}"
    echo "Refusing to start duplicate server. Remove PID file if stale: $PID_FILE"
    exit 1
  fi
fi

echo "Starting vLLM on port ${PORT}..."
echo "  model_path:        ${MODEL_PATH}"
echo "  served_model_name: ${SERVED_MODEL_NAME}"
echo "  cuda_visible:      ${CUDA_VISIBLE_DEVICES}"
echo "  log_file:          ${LOG_FILE}"
echo "  pid_file:          ${PID_FILE}"

nohup env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --port "${PORT}" \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --async-scheduling \
    --enable-chunked-prefill \
    --enable-prefix-caching \
  >"${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"
echo "Started PID=${PID}"

echo "Waiting for health endpoint /v1/models ..."
START_TS="$(date +%s)"
while true; do
  NOW_TS="$(date +%s)"
  ELAPSED="$((NOW_TS - START_TS))"
  if (( ELAPSED > HEALTH_TIMEOUT_SEC )); then
    echo "Timed out after ${HEALTH_TIMEOUT_SEC}s waiting for /v1/models"
    echo "Check log: ${LOG_FILE}"
    exit 1
  fi

  if ! kill -0 "${PID}" 2>/dev/null; then
    echo "vLLM process exited early. Check log: ${LOG_FILE}"
    exit 1
  fi

  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM is ready on http://127.0.0.1:${PORT}"
    echo "Served models:"
    curl -fsS "http://127.0.0.1:${PORT}/v1/models" | jq -r '.data[].id' || true
    echo "Watch logs with:"
    echo "  tail -f ${LOG_FILE}"
    exit 0
  fi

  sleep 1
done

