# scripts_macos/get_piper_generator.sh
#!/usr/bin/env bash
set -euo pipefail

PIPER_REPO_URL="https://github.com/TaterTotterson/piper-sample-generator.git"
download_file() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$out" "$url"
  else
    echo "❌ Need curl or wget to download voice models."
    exit 1
  fi
}

# venv assumed active outside
if [[ ! -d "piper-sample-generator" ]]; then
  echo "⬇️ Cloning TaterTotterson/piper-sample-generator…"
  git clone "$PIPER_REPO_URL" >/dev/null
else
  current_origin="$(git -C piper-sample-generator remote get-url origin 2>/dev/null || true)"
  if [[ "$current_origin" != "$PIPER_REPO_URL" ]]; then
    echo "🔁 Updating piper-sample-generator origin to TaterTotterson fork…"
    git -C piper-sample-generator remote set-url origin "$PIPER_REPO_URL"
  fi
fi

echo "📦 Installing piper-sample-generator in editable mode…"
pip install -q -e ./piper-sample-generator

# Torch/torchaudio for Mac (MPS works out of the box on Apple Silicon wheels)
pip install -q torch torchaudio

# (Kept for phonemization parity with your notebook)
pip install -q piper-phonemize-cross==1.2.1

MODELS_DIR="piper-sample-generator/models"
VOICES_DIR="piper-sample-generator/voices"
mkdir -p "$MODELS_DIR" "$VOICES_DIR"

# English multi-speaker model (used by --language=en)
EN_MODEL_NAME="en_US-libritts_r-medium.pt"
EN_MODEL_URL="https://github.com/TaterTotterson/piper-sample-generator/releases/download/models/${EN_MODEL_NAME}"
if [[ ! -f "${MODELS_DIR}/${EN_MODEL_NAME}" ]]; then
  echo "⬇️ Downloading ${EN_MODEL_NAME}…"
  download_file "${EN_MODEL_URL}" "${MODELS_DIR}/${EN_MODEL_NAME}"
fi
if [[ ! -f "${MODELS_DIR}/${EN_MODEL_NAME}.json" ]]; then
  echo "⬇️ Downloading ${EN_MODEL_NAME}.json…"
  download_file "${EN_MODEL_URL}.json" "${MODELS_DIR}/${EN_MODEL_NAME}.json"
fi

echo "ℹ️ Non-English Piper voices are downloaded on demand for the selected language."

echo "✅ piper-sample-generator ready."
