#!/usr/bin/env bash
set -euo pipefail

# ===============================
# GPT-SoVITS 启动脚本（按 start_gptsovits.ps1 的思路）
# 配置来源：config.txt
# ===============================

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG_FILE="$ROOT_DIR/config.txt"

load_config() {
  local file="$1"
  if [[ ! -f "$file" ]]; then
    echo "❌ config.txt not found: $file" >&2
    exit 1
  fi

  while IFS='=' read -r k v; do
    k="$(echo "${k:-}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    v="$(echo "${v:-}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "$k" ]] && continue
    [[ "$k" == \#* || "$k" == \;* ]] && continue

    # 去掉包裹引号（与 ps1 一致）
    if [[ ( "$v" == "*" && "$v" == *" ) || ( "$v" == '*' && "$v" == *' ) ]]; then
      v="${v:1:${#v}-2}"
    fi

    export "$k=$v"
  done < "$file"
}

path_norm() { echo "$1" | sed 's|\\|/|g'; }

load_config "$CFG_FILE"

# 默认值（与 ps1 一致）
: "${GPTSOVITS_DIR:=GPT-SoVITS}"
: "${GPTSOVITS_API:=GPT-SoVITS/api_v2.py}"
: "${GPTSOVITS_PORT:=9880}"

GPTSOVITS_DIR="$(path_norm "$GPTSOVITS_DIR")"
GPTSOVITS_API="$(path_norm "$GPTSOVITS_API")"

GPT_DIR="$ROOT_DIR/$GPTSOVITS_DIR"
PYTHON_EXE="$GPT_DIR/runtime/python"
API_SCRIPT="$ROOT_DIR/$GPTSOVITS_API"

# 解析/补全路径（与 ps1 一致）
if [[ -n "${TTS_REF_AUDIO_PATH:-}" ]]; then
  if command -v realpath >/dev/null 2>&1; then
    export TTS_REF_AUDIO_PATH="$(realpath "$ROOT_DIR/$TTS_REF_AUDIO_PATH")"
  elif command -v readlink >/dev/null 2>&1; then
    export TTS_REF_AUDIO_PATH="$(readlink -f "$ROOT_DIR/$TTS_REF_AUDIO_PATH" 2>/dev/null || echo "$ROOT_DIR/$TTS_REF_AUDIO_PATH")"
  else
    export TTS_REF_AUDIO_PATH="$ROOT_DIR/$TTS_REF_AUDIO_PATH"
  fi
fi

if [[ -z "${GPTSOVITS_URL:-}" ]]; then
  export GPTSOVITS_URL="http://127.0.0.1:${GPTSOVITS_PORT}"
fi

echo "==========================================="
echo "GPT-SoVITS Config (from config.txt)"
echo "RootDir : $ROOT_DIR"
echo "GptDir  : $GPT_DIR"
echo "Python  : $PYTHON_EXE"
echo "API     : $API_SCRIPT"
echo "URL     : $GPTSOVITS_URL"
echo "==========================================="

if [[ ! -d "$GPT_DIR" ]]; then
  echo "❌ GPTSOVITS_DIR not found: $GPT_DIR" >&2
  exit 1
fi
if [[ ! -f "$API_SCRIPT" ]]; then
  echo "❌ api_v2.py not found: $API_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_EXE" ]]; then
  echo "❌ runtime/python not found (or not executable): $PYTHON_EXE" >&2
  exit 1
fi

cd "$GPT_DIR"

# 与 ps1 一致：把 runtime 放到 PATH 里（有些依赖会找 dll/so 或可执行文件）
export PATH="$GPT_DIR/runtime:$PATH"

# 兼容：一些场景需要 PYTHONPATH 指向工程目录（原 sh 已设置）
export PYTHONPATH="$GPT_DIR"

# 组装额外参数（与 ps1 一致：按空格切分）
ARGS=()
if [[ -n "${GPTSOVITS_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS=($GPTSOVITS_ARGS)
fi

if [[ "${GPTSOVITS_REDIRECT_LOG:-0}" == "1" ]]; then
  LOG_DIR="$ROOT_DIR/logs"
  mkdir -p "$LOG_DIR"
  TS="$(date +"%Y%m%d_%H%M%S")"
  LOG_FILE="$LOG_DIR/gptsovits_${TS}.log"

  # ps1 使用 *>> 追加；bash 用 >> 2>&1
  exec "$PYTHON_EXE" -I "$API_SCRIPT" "${ARGS[@]}" >>"$LOG_FILE" 2>&1
else
  exec "$PYTHON_EXE" -I "$API_SCRIPT" "${ARGS[@]}"
fi
