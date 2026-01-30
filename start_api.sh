#!/usr/bin/env bash
# ===============================
# start_api.sh (aligned with start_api.ps1 logic)
# ===============================
set -euo pipefail

# -------- 0) 计算根目录 --------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# -------- 1) 激活虚拟环境 --------
VENV_ACTIVATE="$ROOT_DIR/venv/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
  echo "[API] venv activated: ${VIRTUAL_ENV:-}"
else
  echo "[API] venv not found: $VENV_ACTIVATE" >&2
  exit 1
fi

# -------- 2) Load config --------
CFG_FILE="$ROOT_DIR/config.txt"
if [[ ! -f "$CFG_FILE" ]]; then
  echo "[API] config.txt not found: $CFG_FILE" >&2
  exit 1
fi

# Trim helper
_trim() {
  local s="${1:-}"
  # remove leading/trailing whitespace
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

# Normalize path (Windows \ -> /)
path_norm() { echo "${1:-}" | sed 's|\\|/|g'; }

# Read config into env (only KEY=VALUE lines; ignore blank/#)
while IFS= read -r line || [[ -n "$line" ]]; do
  line="$(_trim "$line")"
  [[ -z "$line" ]] && continue
  [[ "$line" == \#* ]] && continue
  if [[ "$line" == *"="* ]]; then
    key="$(_trim "${line%%=*}")"
    val="$(_trim "${line#*=}")"
    [[ -z "$key" ]] && continue
    export "$key=$val"
  fi
done < "$CFG_FILE"

# -------- 3) 导出环境变量 --------
# (ps1 显式导出这几个；这里保持一致)
export API_HOST="${API_HOST:-0.0.0.0}"
export API_PORT="${API_PORT:-8000}"
export API_RELOAD="${API_RELOAD:-}"
export API_LOG_LEVEL="${API_LOG_LEVEL:-}"

# -------- 4) 启动 FastAPI --------
API_ENTRY="$(path_norm "${API_ENTRY:-api_server.py}")"
if [[ ! -f "$ROOT_DIR/$API_ENTRY" && ! -f "$API_ENTRY" ]]; then
  echo "[API] API_ENTRY not found: $API_ENTRY" >&2
  exit 1
fi

echo "[API] Starting FastAPI on ${API_HOST}:${API_PORT}"
python "$API_ENTRY"
