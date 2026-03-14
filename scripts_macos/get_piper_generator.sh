# scripts_macos/get_piper_generator.sh
#!/usr/bin/env bash
set -euo pipefail

PIPER_REPO_URL="https://github.com/TaterTotterson/piper-sample-generator.git"

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

echo "✅ piper-sample-generator ready."
