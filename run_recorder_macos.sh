#!/usr/bin/env bash
# run_recorder_macos.sh
#
# One-command launcher for the local mic recorder UI:
# - creates/uses .recorder-venv
# - installs pinned dependencies once
# - runs uvicorn from the venv (never uses global /opt/homebrew/bin/uvicorn)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${REC_VENV_DIR:-$ROOT_DIR/.recorder-venv}"
PY="$VENV_DIR/bin/python"
PIP="$PY -m pip"
PIN_FILE="$VENV_DIR/.pinned_installed"

# Optional host/port
HOST="${REC_HOST:-127.0.0.1}"
PORT="${REC_PORT:-8789}"

# Pinned deps (edit if you want)
FASTAPI_VERSION="${REC_FASTAPI_VERSION:-0.115.6}"
UVICORN_VERSION="${REC_UVICORN_VERSION:-0.30.6}"
PY_MULTIPART_VERSION="${REC_PY_MULTIPART_VERSION:-0.0.9}"

echo "🎙️ microWakeWord Recorder (local)"
echo "→ ROOT: $ROOT_DIR"
echo "→ VENV: $VENV_DIR"

# Create venv if missing
if [[ ! -x "$PY" ]]; then
  echo "🧹 Creating recorder venv: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# Always activate (handy for PATH vars), but we still use $PY explicitly
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install pinned deps once
if [[ ! -f "$PIN_FILE" ]]; then
  echo "🧹 Fresh recorder venv → installing pinned dependencies"
  $PIP install -U pip setuptools wheel

  # Core server deps
  $PIP install \
    "fastapi==${FASTAPI_VERSION}" \
    "uvicorn[standard]==${UVICORN_VERSION}" \
    "python-multipart==${PY_MULTIPART_VERSION}"

  touch "$PIN_FILE"
else
  echo "✅ Reusing existing .recorder-venv (no upgrades)"
fi

# HARD FAIL: ensure pip is the venv pip
VENV_PREFIX="$("$PY" -c 'import sys; print(sys.prefix)')"
$PIP -V | grep -q "$VENV_PREFIX" || {
  echo "❌ pip is not using venv ($VENV_PREFIX)"
  $PIP -V
  exit 1
}

# HARD FAIL: ensure uvicorn is the venv one
UVICORN="$VENV_DIR/bin/uvicorn"
if [[ ! -x "$UVICORN" ]]; then
  echo "❌ uvicorn not found in venv: $UVICORN"
  echo "   Try: $PIP install 'uvicorn[standard]'"
  exit 1
fi

echo "→ Launching: $UVICORN recorder_server:app --host $HOST --port $PORT"
exec "$UVICORN" recorder_server:app --host "$HOST" --port "$PORT"