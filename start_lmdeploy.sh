#!/usr/bin/env bash
set -euo pipefail

# ===============================
# start_lmdeploy.sh (aligned to start_lmdeploy.ps1)
# ===============================

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# ---------- 0) activate venv ----------
VENV_ACTIVATE="$ROOT_DIR/venv/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
  echo "[LMDeploy] venv activated: ${VIRTUAL_ENV:-}"
else
  echo "[LMDeploy] venv not found: $VENV_ACTIVATE" >&2
  exit 1
fi

# ---------- 1) CUDA_PATH: prefer local cuda_stub ----------
CUDA_STUB="$ROOT_DIR/cuda_stub"
if [[ -z "${CUDA_PATH:-}" ]]; then
  if [[ -d "$CUDA_STUB" ]]; then
    export CUDA_PATH="$CUDA_STUB"
  fi
fi
if [[ -n "${CUDA_PATH:-}" ]]; then
  if [[ -d "$CUDA_PATH/bin" ]]; then
    export PATH="$CUDA_PATH/bin:$PATH"
  fi
  echo "[LMDeploy] CUDA_PATH: $CUDA_PATH"
fi

CFG_FILE="$ROOT_DIR/config.txt"

# ---------- 2) load config.txt into environment ----------
load_config() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "[LMDeploy] config.txt not found: $file" >&2
    exit 1
  fi
  while IFS='=' read -r k v; do
    k="$(echo "${k:-}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    v="$(echo "${v:-}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "$k" ]] && continue
    [[ "$k" == \#* || "$k" == \;* ]] && continue

    # strip wrapping quotes (match ps1 behavior)
    if [[ ( "$v" == "*" && "$v" == *" ) || ( "$v" == '*' && "$v" == *' ) ]]; then
      v="${v:1:${#v}-2}"
    fi

    export "$k=$v"
  done < "$file"
}

# normalize windows path to unix path
path_norm() { echo "$1" | sed 's|\\|/|g'; }

load_config "$CFG_FILE"

# ---------- 3) read params with defaults ----------
SUBCMD="${LMDEPLOY_SUBCMD:-serve api_server}"
MODEL_REL="$(path_norm "${LMDEPLOY_MODEL_PATH:-}")"
BACKEND="${LMDEPLOY_BACKEND:-turbomind}"
MODEL_FORMAT="${LMDEPLOY_MODEL_FORMAT:-}"
MODEL_NAME="${LMDEPLOY_MODEL_NAME:-}"
SERVER_NAME="${LMDEPLOY_SERVER_NAME:-0.0.0.0}"
SERVER_PORT="${LMDEPLOY_SERVER_PORT:-23333}"
TP="${LMDEPLOY_TP:-}"
SESSION_LEN="${LMDEPLOY_SESSION_LEN:-}"
CACHE_MAX_ENTRY="${LMDEPLOY_CACHE_MAX_ENTRY_COUNT:-}"
LOG_LEVEL="${LMDEPLOY_LOG_LEVEL:-WARNING}"

# ---------- 4) force local model path ----------
if [[ -z "$MODEL_REL" ]]; then
  echo "[LMDeploy] LMDEPLOY_MODEL_PATH is empty in config.txt" >&2
  exit 1
fi

MODEL_PATH="$MODEL_REL"
# if not absolute, anchor to ROOT_DIR
if [[ "$MODEL_PATH" != /* ]]; then
  MODEL_PATH="$ROOT_DIR/$MODEL_PATH"
fi

echo "==========================================="
echo "[LMDeploy] RootDir      : $ROOT_DIR"
echo "[LMDeploy] Subcmd       : $SUBCMD"
echo "[LMDeploy] ModelPath    : $MODEL_PATH"
echo "[LMDeploy] Backend      : $BACKEND"
echo "[LMDeploy] ModelFormat  : $MODEL_FORMAT"
echo "[LMDeploy] ModelName    : $MODEL_NAME"
echo "[LMDeploy] ServerName   : $SERVER_NAME"
echo "[LMDeploy] ServerPort   : $SERVER_PORT"
echo "[LMDeploy] TP           : $TP"
echo "[LMDeploy] SessionLen   : $SESSION_LEN"
echo "[LMDeploy] CacheMaxEntry: $CACHE_MAX_ENTRY"
echo "[LMDeploy] LogLevel     : $LOG_LEVEL"
echo "==========================================="

# block online download: if local dir missing, fail fast
if [[ ! -e "$MODEL_PATH" ]]; then
  echo "[LMDeploy] 本地模型路径不存在（将不会尝试在线下载）：$MODEL_PATH" >&2
  if [[ -d "$ROOT_DIR/model" ]]; then
    echo "[Hint] 当前 ./model/ 下的条目："
    ls -1 "$ROOT_DIR/model" | sed 's/^/  - /'
  fi
  exit 1
fi

# ---------- 5) offline env (recommended) ----------
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"

# ---------- 6) build lmdeploy args ----------
# split subcmd by spaces into array
read -r -a SUBPARTS <<< "$SUBCMD"

ARGS=()
ARGS+=("${SUBPARTS[@]}")
ARGS+=("$MODEL_PATH")
ARGS+=("--server-name" "$SERVER_NAME" "--server-port" "$SERVER_PORT")

[[ -n "$BACKEND" ]]       && ARGS+=("--backend" "$BACKEND")
[[ -n "$MODEL_FORMAT" ]]  && ARGS+=("--model-format" "$MODEL_FORMAT")
[[ -n "$MODEL_NAME" ]]    && ARGS+=("--model-name" "$MODEL_NAME")
[[ -n "$TP" ]]            && ARGS+=("--tp" "$TP")
[[ -n "$SESSION_LEN" ]]   && ARGS+=("--session-len" "$SESSION_LEN")
[[ -n "$CACHE_MAX_ENTRY" ]] && ARGS+=("--cache-max-entry-count" "$CACHE_MAX_ENTRY")
[[ -n "$LOG_LEVEL" ]]     && ARGS+=("--log-level" "$LOG_LEVEL")

echo "[LMDeploy] Command: lmdeploy ${ARGS[*]}"

# ---------- 7) start ----------
exec lmdeploy "${ARGS[@]}"
