#!/usr/bin/env bash
# train_microwakeword_macos.sh
# One-shot: setup (idempotent) + run pipeline on Apple Silicon (macOS).
# Usage:
#   ./train_microwakeword_macos.sh "hey_tater" 50000 100 \
#       --piper-model /path/to/voice1.onnx --piper-model /path/to/voice2.pt
#
# If no --piper-model is given, we auto-download a default .pt voice.

set -euo pipefail

TARGET_WORD="${1:-hey_tater}"
MAX_TTS_SAMPLES="${2:-50000}"
BATCH_SIZE="${3:-100}"
[[ $# -ge 3 ]] && shift 3 || shift $#

# Collect any --piper-model flags (repeatable)
PIPER_MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --piper-model) PIPER_MODELS+=("$2"); shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "❌ This script is intended for macOS (Apple Silicon)."; exit 1
fi

# ── Ensure system deps ────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  echo "❌ Homebrew is required but not found. Install from https://brew.sh/ first."
  exit 1
fi

# brew is not installing ffmpeg if version not specified. libtorchcodec is not yet compatible with ffmpeg v8
echo "📦 Ensuring ffmpeg@7 + wget are installed (via Homebrew)…"
brew list ffmpeg@7 &>/dev/null || brew install ffmpeg@7
brew list wget &>/dev/null || brew install wget

# Workaround: on some macOS (e.g. M4 / Tahoe), torchcodec fails to locate ffmpeg libs
FFMPEG_LIB_DIR="$(brew --prefix ffmpeg@7)/lib"
if [[ -d "$FFMPEG_LIB_DIR" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$FFMPEG_LIB_DIR:${DYLD_FALLBACK_LIBRARY_PATH:-}"
  echo "✅ ffmpeg library path set: $FFMPEG_LIB_DIR"
fi

# ── venv (force ARM64 & a known-good Python) ──────────────────────────────────
# You can override PYTHON_BIN if you prefer a different path/version.
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.11}"

if [[ ! -d ".venv" ]]; then
  echo "🧪 Creating ARM64 venv with $PYTHON_BIN"
  arch -arm64 "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Sanity prints (helpful if TF wheels fail)
echo "python: $(which python)"
echo "pip:    $(which pip)"
python - <<'PY'
import platform, sys
print("Python:", sys.version.replace("\n"," "))
print("Arch:  ", platform.machine())
PY

# Ensure we’re on arm64 + supported Python (3.10/3.11)
ARCH=$(python -c 'import platform; print(platform.machine())')
PYVER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if [[ "$ARCH" != "arm64" ]]; then
  echo "❌ venv arch is $ARCH (needs arm64). Recreate with: rm -rf .venv && arch -arm64 $PYTHON_BIN -m venv .venv"
  exit 1
fi
case "$PYVER" in
  3.10|3.11) : ;; # ok
  *) echo "❌ Detected Python $PYVER. Use 3.10 or 3.11 for tensorflow-macos."
     echo "   Try: brew install python@3.11 && rm -rf .venv && arch -arm64 /opt/homebrew/bin/python3.11 -m venv .venv"
     exit 1 ;;
esac

# ── deps (idempotent; always use the venv's pip via python -m pip) ────────────
python -m pip install -U pip setuptools wheel >/dev/null

# --- versions (override at runtime if needed) ---
TF_VERSION="${TF_VERSION:-2.16.2}"
TF_METAL_VERSION="${TF_METAL_VERSION:-1.2.0}"

# Ensure supported Python + arch (TF 2.16 has wheels for 3.10–3.12 on arm64)
ARCH=$(python -c 'import platform; print(platform.machine())')
PYVER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$ARCH" != "arm64" ]]; then
  echo "❌ venv arch is $ARCH; needs arm64."; exit 1
fi
case "$PYVER" in
  3.10|3.11|3.12) : ;; 
  *) echo "❌ Python $PYVER detected. Use 3.10–3.12 for TF ${TF_VERSION}."; exit 1 ;;
esac

# Install pinned TensorFlow + Metal
python -m pip install -q "tensorflow-macos==${TF_VERSION}" "tensorflow-metal==${TF_METAL_VERSION}"

# Self-heal if plugin load fails (common when versions get out of sync)
python - <<'PY'
import sys, traceback, subprocess
try:
    import tensorflow as tf
    print("✅ TensorFlow", tf.__version__, "loaded.")
except Exception as e:
    print("⚠️  TensorFlow failed to load, attempting self-heal…")
    traceback.print_exc()
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y",
                           "tensorflow-metal", "tensorflow-macos", "tensorflow"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                           "--force-reinstall", f"tensorflow-macos=={os.environ.get('TF_VERSION','2.16.2')}"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir",
                           "--force-reinstall", f"tensorflow-metal=={os.environ.get('TF_METAL_VERSION','1.2.0')}"])
    import tensorflow as tf
    print("✅ TensorFlow", tf.__version__, "recovered.")
print("Devices:", [d.name for d in __import__("tensorflow").config.list_logical_devices()])
PY


# Other deps
python -m pip install -q "git+https://github.com/puddly/pymicro-features@puddly/minimum-cpp-version" \
                           "git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f" || true
# Audio/ML stack
python -m pip install -q datasets soundfile librosa scipy numpy tqdm pyyaml requests ipython jupyter || true

# needed for HF Audio decoding in streaming mode
python -m pip install -q torchcodec || true


# microWakeWord source (editable)
if [[ ! -d "micro-wake-word" ]]; then
  echo "⬇️ Cloning microWakeWord…"
  git clone https://github.com/TaterTotterson/micro-wake-word.git >/dev/null
else
  echo "🔁 Updating microWakeWord…"
  (cd micro-wake-word && git pull --ff-only origin main || true)
fi

python -m pip install -q -e ./micro-wake-word || true
# Official piper-sample-generator (replaces fork)
bash scripts_macos/get_piper_generator.sh

# ── verify Metal GPU (optional) ───────────────────────────────────────────────
python - <<'PY'
import tensorflow as tf
devs = tf.config.list_logical_devices()
print("✅ TF logical devices:", [d.name for d in devs])
if not any("GPU" in d.device_type for d in devs):
    print("⚠️  No GPU logical device detected. Will run on CPU.")
PY

# ── export for inline python ──────────────────────────────────────────────────
export TARGET_WORD MAX_TTS_SAMPLES BATCH_SIZE

# ── Ensure at least one model is provided (auto-fetch default if none) ────────
DEFAULT_MODEL_PT="piper-sample-generator/models/en_US-libritts_r-medium.pt"
if [[ ${#PIPER_MODELS[@]} -eq 0 ]]; then
  echo "ℹ️  No --piper-model provided; using default voice:"
  echo "    $DEFAULT_MODEL_PT"
  mkdir -p "$(dirname "$DEFAULT_MODEL_PT")"
  if [[ ! -f "$DEFAULT_MODEL_PT" ]]; then
    wget -q -O "$DEFAULT_MODEL_PT" \
      "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
  fi
  PIPER_MODELS=("$DEFAULT_MODEL_PT")
fi


# ── Pass models to Python via env ─────────────────────────────────────────────
PIPER_MODELS_CSV=""
if [[ ${#PIPER_MODELS[@]} -gt 0 ]]; then
  PIPER_MODELS_CSV=$(IFS=,; echo "${PIPER_MODELS[*]}")
fi
export PIPER_MODELS_CSV

# ── (A) clean previous run artifacts (match NVIDIA/Streamlit version) ────────
echo "🧹 Cleaning previous run artifacts…"
rm -f  training_parameters.yaml
rm -rf trained_models
rm -rf generated_augmented_features
rm -rf generated_samples
echo "✅ Cleanup done."

# make sure the folder exists so 'find' won't fail under 'set -e'
mkdir -p generated_samples

# ── (B) bulk TTS (skip if enough files present) ──────────────────────────────
count_existing=$(find generated_samples -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
if [[ "${count_existing:-0}" -lt "$MAX_TTS_SAMPLES" ]]; then
  echo "🎤 Generating ${MAX_TTS_SAMPLES} samples for '${TARGET_WORD}' (batch ${BATCH_SIZE})…"
  python - <<'PY'
import os, sys, shlex, subprocess

# make sure the generator is importable
if "piper-sample-generator/" not in sys.path:
    sys.path.append("piper-sample-generator/")

models = [m.strip() for m in os.environ.get("PIPER_MODELS_CSV","").split(",") if m.strip()]
model_flags = sum([["--model", m] for m in models], [])

cmd = [
    sys.executable,
    "piper-sample-generator/generate_samples.py",
    os.environ["TARGET_WORD"],
    "--max-samples", os.environ["MAX_TTS_SAMPLES"],
    "--batch-size",  os.environ["BATCH_SIZE"],
    "--output-dir",  "generated_samples",
    *model_flags,
]

print("CMD:", " ".join(shlex.quote(c) for c in cmd))
proc = subprocess.run(cmd, text=True, capture_output=False)
if proc.returncode != 0:
    raise SystemExit(proc.returncode)
PY
else
  echo "✅ Found ${count_existing} samples (>= desired); skipping TTS generation."
fi

# ── (C) pull/prepare augmentation datasets (RIR, Audioset, FMA) ──────────────
python scripts_macos/prepare_datasets.py

# ── (D) build augmenter + spectrogram feature mmaps ───────────────────────────
python scripts_macos/make_features.py

# ── (E) download precomputed negative spectrograms ────────────────────────────
python scripts_macos/fetch_negatives.py

# ── (F) write training YAML (tuned for your notebook) ────────────────────────
python scripts_macos/write_training_yaml.py

# ── (G) train + export (Metal TF) ────────────────────────────────────────────
python -m microwakeword.model_train_eval \
  --training_config=training_parameters.yaml \
  --train 1 \
  --restore_checkpoint 1 \
  --test_tf_nonstreaming 0 \
  --test_tflite_nonstreaming 0 \
  --test_tflite_nonstreaming_quantized 0 \
  --test_tflite_streaming 0 \
  --test_tflite_streaming_quantized 1 \
  --use_weights "best_weights" \
  mixednet \
  --pointwise_filters "64,64,64,64" \
  --repeat_in_block "1,1,1,1" \
  --mixconv_kernel_sizes "[5], [7,11], [9,15], [23]" \
  --residual_connection "0,0,0,0" \
  --first_conv_filters 32 \
  --first_conv_kernel_size 5 \
  --stride 2

# ── (H) package artifacts (name by wake word) ─────────────────────────────────
python - <<'PY'
import os, re, shutil, json
from pathlib import Path

# 1) derive a safe base name from TARGET_WORD (fallback "wakeword")
target = os.environ.get("TARGET_WORD", "wakeword")
# normalize: lowercase, spaces->underscore, strip invalids
safe = re.sub(r'[^a-z0-9_]+', '', re.sub(r'\s+', '_', target.lower()))
if not safe:
    safe = "wakeword"

# 2) source tflite (from training export) and destination
src = Path("trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite")
dst = Path(f"{safe}.tflite")
if not src.exists():
    raise SystemExit(f"❌ Model not found at {src}")
shutil.copy(src, dst)

# 3) write JSON metadata pointing to the renamed model
meta = {
  "type": "micro",
  "wake_word": target,
  "author": "Tater Totterson",
  "website": "https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon",
  "model": f"{safe}.tflite",
  "trained_languages": ["en"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.97,
    "sliding_window_size": 5,
    "feature_step_size": 10,
    "tensor_arena_size": 30000,
    "minimum_esphome_version": "2024.7.0"
  }
}
json_path = Path(f"{safe}.json")
json_path.write_text(json.dumps(meta, indent=2))

print(f"📦 Wrote {dst.name} and {json_path.name} (wake word: {target!r})")
PY

echo "🎉 Done."