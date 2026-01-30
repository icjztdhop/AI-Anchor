#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG_FILE="$ROOT_DIR/config.txt"

if [ ! -f "$CFG_FILE" ]; then
  echo "[ERROR] config.txt not found in: $ROOT_DIR"
  exit 1
fi

echo "========================================="
echo " Virtual Anchor - Start All (bash)"
echo "========================================="

# ------------------------------------------
# Load config.txt -> env
# ------------------------------------------
set -a
while IFS='=' read -r key value; do
  key="$(echo "$key" | sed 's/[[:space:]]//g')"
  value="$(echo "$value" | sed 's/^[[:space:]]*//g')"
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  export "$key=$value"
done < "$CFG_FILE"
set +a

LM_PID=""
TTS_PID=""

cleanup() {
  echo ""
  echo "Stopping services..."
  if [ -n "${LM_PID:-}" ] && kill -0 "$LM_PID" >/dev/null 2>&1; then
    kill "$LM_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${TTS_PID:-}" ] && kill -0 "$TTS_PID" >/dev/null 2>&1; then
    kill "$TTS_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

# ------------------------------------------
# 1) Start LMDeploy (background)
# ------------------------------------------
echo "[1/3] Starting LMDeploy..."
(
  cd "$ROOT_DIR"
  lmdeploy     $LMDEPLOY_SUBCMD     "$LMDEPLOY_MODEL_PATH"     --backend "$LMDEPLOY_BACKEND"     --model-format "$LMDEPLOY_MODEL_FORMAT"     --model-name "$LMDEPLOY_MODEL_NAME"     --server-name "$LMDEPLOY_SERVER_NAME"     --server-port "$LMDEPLOY_SERVER_PORT"     --tp "$LMDEPLOY_TP"     --session-len "$LMDEPLOY_SESSION_LEN"     --cache-max-entry-count "$LMDEPLOY_CACHE_MAX_ENTRY_COUNT"
) > "$ROOT_DIR/lmdeploy.log" 2>&1 &
LM_PID=$!
sleep 5

# ------------------------------------------
# 2) Start GPT-SoVITS (background)
# ------------------------------------------
echo "[2/3] Starting GPT-SoVITS..."
(
  cd "$ROOT_DIR/$GPTSOVITS_DIR"
  python "$GPTSOVITS_API"     --host "$GPTSOVITS_HOST"     --port "$GPTSOVITS_PORT"     $GPTSOVITS_ARGS
) > "$ROOT_DIR/gptsovits.log" 2>&1 &
TTS_PID=$!
sleep 5

# ------------------------------------------
# 3) Start API Server (foreground)
# ------------------------------------------
echo "[3/3] Starting API Server..."
cd "$ROOT_DIR"
python "$API_ENTRY"
