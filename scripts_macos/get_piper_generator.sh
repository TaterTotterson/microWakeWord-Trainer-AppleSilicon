# scripts_macos/get_piper_generator.sh
#!/usr/bin/env bash
set -euo pipefail

# venv assumed active outside
if [[ ! -d "piper-sample-generator" ]]; then
  echo "⬇️ Cloning rhasspy/piper-sample-generator…"
  git clone https://github.com/rhasspy/piper-sample-generator.git >/dev/null
fi

echo "📦 Installing piper-sample-generator in editable mode…"
pip install -q -e ./piper-sample-generator

# Torch/torchaudio for Mac (MPS works out of the box on Apple Silicon wheels)
pip install -q torch torchaudio

# (Kept for phonemization parity with your notebook)
pip install -q piper-phonemize-cross==1.2.1

echo "✅ piper-sample-generator ready."