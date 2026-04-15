#!/usr/bin/env bash
# train_microwakeword_macos.sh
# One-shot: setup (idempotent) + run pipeline on Apple Silicon (macOS).
# Usage:
#   ./train_microwakeword_macos.sh "hey_tater" 50000 100 \
#       --language en \
#       --piper-model /path/to/voice1.onnx --piper-model /path/to/voice2.pt
#
# If no --piper-model is given, we use language-aware defaults.

set -euo pipefail

TARGET_WORD="${1:-hey_tater}"
MAX_TTS_SAMPLES="${2:-50000}"
BATCH_SIZE="${3:-100}"
[[ $# -ge 3 ]] && shift 3 || shift $#

# Default language can be overridden by --language or MWW_LANGUAGE
LANGUAGE="${MWW_LANGUAGE:-en}"

# Collect optional flags
PIPER_MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --piper-model) PIPER_MODELS+=("$2"); shift 2 ;;
    --language) LANGUAGE="${2:-}"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

LANGUAGE="$(echo "${LANGUAGE}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "$LANGUAGE" ]]; then
  LANGUAGE="en"
fi
export MWW_LANGUAGE="$LANGUAGE"
echo "🌐 Training language: $LANGUAGE"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "❌ This script is intended for macOS (Apple Silicon)."; exit 1
fi

# ── Ensure system deps ────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  echo "❌ Homebrew is required but not found. Install from https://brew.sh/ first."
  exit 1
fi

echo "📦 Ensuring ffmpeg@7 + wget are installed (via Homebrew)…"

# wget first
brew list wget &>/dev/null || brew install wget

# prefer ffmpeg@7 because torchcodec wants < 8
if brew info ffmpeg@7 &>/dev/null; then
  brew list ffmpeg@7 &>/dev/null || brew install ffmpeg@7
  FFMPEG_PREFIX="$(brew --prefix ffmpeg@7)"
  echo "✅ Using ffmpeg@7 at $FFMPEG_PREFIX"
else
  # fallback if ffmpeg@7 isn’t available on this Homebrew
  brew list ffmpeg &>/dev/null || brew install ffmpeg
  FFMPEG_PREFIX="$(brew --prefix ffmpeg)"
  echo "⚠️ ffmpeg@7 not found; using default ffmpeg instead"
fi

# Make the chosen ffmpeg visible to torchcodec on macOS (ARM sometimes needs DYLD_*)
FFMPEG_LIB_DIR="$FFMPEG_PREFIX/lib"
if [[ -d "$FFMPEG_LIB_DIR" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="$FFMPEG_LIB_DIR:${DYLD_FALLBACK_LIBRARY_PATH:-}"
  export DYLD_LIBRARY_PATH="$FFMPEG_LIB_DIR:${DYLD_LIBRARY_PATH:-}"
  echo "✅ ffmpeg library path set: $FFMPEG_LIB_DIR"
else
  echo "⚠️ Could not find ffmpeg lib dir at $FFMPEG_LIB_DIR"
fi

# ── venv (ARM64 + pinned stack, install once) ────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.11}"

TF_VERSION="${TF_VERSION:-2.16.2}"
TF_METAL_VERSION="${TF_METAL_VERSION:-1.2.0}"
KERAS_VERSION="${KERAS_VERSION:-3.3.3}"
PROTOBUF_VERSION="${PROTOBUF_VERSION:-4.25.8}"
FLATBUFFERS_VERSION="${FLATBUFFERS_VERSION:-23.5.26}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"

if [[ ! -d ".venv" ]]; then
  echo "🧪 Creating ARM64 venv with $PYTHON_BIN"
  arch -arm64 "$PYTHON_BIN" -m venv .venv
fi

# always activate (both create + reuse)
# shellcheck disable=SC1091
source .venv/bin/activate

# canonical python for the rest of the script (never rely on PATH again)
PY="$(pwd)/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "❌ venv python not found at $PY"
  exit 1
fi

if [[ ! -f ".venv/.pinned_installed" ]]; then
  echo "🧹 Fresh venv → installing pinned toolchain"
  "$PY" -m pip install -U pip setuptools wheel

  # Pinned TF/Keras stack (stable)
  "$PY" -m pip install \
    "protobuf==${PROTOBUF_VERSION}" \
    "flatbuffers==${FLATBUFFERS_VERSION}" \
    "keras==${KERAS_VERSION}" \
    "tensorflow-macos==${TF_VERSION}" \
    "tensorflow-metal==${TF_METAL_VERSION}"

  # Pinned torch stack for torchcodec / datasets audio backend
  "$PY" -m pip install "torch==${TORCH_VERSION}" torchcodec

  touch ".venv/.pinned_installed"
else
  echo "✅ Reusing existing .venv (no upgrades)"
fi

# ── HARD FAIL: ensure pip is the venv pip ────────────────────────────────────
VENV_PREFIX="$("$PY" -c 'import sys; print(sys.prefix)')"
"$PY" -m pip -V | grep -q "$VENV_PREFIX" || {
  echo "❌ pip is not using venv ($VENV_PREFIX)"
  "$PY" -m pip -V
  exit 1
}

# ── Sanity prints ────────────────────────────────────────────────────────────
echo "python: $PY"
echo "pip:    $("$PY" -m pip -V | awk '{print $1, $2, $3, $4, $5}')"
"$PY" - <<'PY'
import platform, sys
print("Python:", sys.version.replace("\n"," "))
print("Arch:  ", platform.machine())
PY

# ── Ensure we’re on arm64 + supported Python ─────────────────────────────────
ARCH=$("$PY" -c 'import platform; print(platform.machine())')
PYVER=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

if [[ "$ARCH" != "arm64" ]]; then
  echo "❌ venv arch is $ARCH (needs arm64). Recreate with:"
  echo "   rm -rf .venv && arch -arm64 $PYTHON_BIN -m venv .venv"
  exit 1
fi
case "$PYVER" in
  3.10|3.11) : ;;
  *) echo "❌ Detected Python $PYVER. Use 3.10 or 3.11 for tensorflow-macos."
     exit 1 ;;
esac

# ── HARD FAIL: verify pinned versions (no silent drift) ──────────────────────
"$PY" - <<PY
import sys
import tensorflow as tf
import keras
import google.protobuf
import flatbuffers

expected = {
  "tensorflow": "${TF_VERSION}",
  "keras": "${KERAS_VERSION}",
  "protobuf": "${PROTOBUF_VERSION}",
  "flatbuffers": "${FLATBUFFERS_VERSION}",
}

actual = {
  "tensorflow": tf.__version__,
  "keras": keras.__version__,
  "protobuf": google.protobuf.__version__,
  "flatbuffers": flatbuffers.__version__,
}

bad = [(k, actual[k], expected[k]) for k in expected if actual[k] != expected[k]]
if bad:
  print("❌ Version drift detected:")
  for k,a,e in bad:
    print(f"  - {k}: {a} (expected {e})")
  print("\nFix by rebuilding venv:")
  print("  rm -rf .venv && arch -arm64 ${PYTHON_BIN} -m venv .venv && ./train_microwakeword_macos.sh ...")
  sys.exit(1)

print("✅ Pinned ML stack verified.")
PY

# tell HF to use torch backend, not soundfile
export DATASETS_AUDIO_BACKEND=torch

# Other deps (best-effort)
"$PY" -m pip install -q "git+https://github.com/puddly/pymicro-features@puddly/minimum-cpp-version" \
                           "git+https://github.com/whatsnowplaying/audio-metadata@d4ebb238e6a401bb1a5aaaac60c9e2b3cb30929f" || true
"$PY" -m pip install -q datasets librosa scipy numpy tqdm pyyaml requests ipython jupyter silero-vad || true

# microWakeWord source (editable)
if [[ ! -d "micro-wake-word" ]]; then
  echo "⬇️ Cloning microWakeWord…"
  git clone https://github.com/TaterTotterson/micro-wake-word.git >/dev/null
else
  echo "🔁 Updating microWakeWord…"
  (cd micro-wake-word && git pull --ff-only origin main || true)
fi

"$PY" -m pip install -q -e ./micro-wake-word || true

# piper-sample-generator (TaterTotterson fork)
bash scripts_macos/get_piper_generator.sh

# ── verify Metal GPU (optional) ───────────────────────────────────────────────
"$PY" - <<'PY'
import tensorflow as tf
devs = tf.config.list_logical_devices()
print("✅ TF logical devices:", [d.name for d in devs])
if not any(d.device_type == "GPU" for d in devs):
    print("⚠️  No GPU logical device detected. Will run on CPU.")
PY

# ── export for inline python ──────────────────────────────────────────────────
export TARGET_WORD MAX_TTS_SAMPLES BATCH_SIZE LANGUAGE MWW_LANGUAGE

# ── Ensure at least one model is provided (language-aware default) ────────────
DEFAULT_MODEL_PT="piper-sample-generator/models/en_US-libritts_r-medium.pt"
if [[ ${#PIPER_MODELS[@]} -eq 0 ]]; then
  if [[ "$LANGUAGE" == "en" ]]; then
    echo "ℹ️  No --piper-model provided; using default English voice:"
    echo "    $DEFAULT_MODEL_PT"
    mkdir -p "$(dirname "$DEFAULT_MODEL_PT")"
    if [[ ! -f "$DEFAULT_MODEL_PT" ]]; then
      wget -q -O "$DEFAULT_MODEL_PT" \
        "https://github.com/TaterTotterson/piper-sample-generator/releases/download/models/en_US-libritts_r-medium.pt"
    fi
    PIPER_MODELS=("$DEFAULT_MODEL_PT")
  else
    shopt -s nullglob
    language_voice_models=(piper-sample-generator/voices/"${LANGUAGE}"_*.onnx)
    shopt -u nullglob
    if [[ ${#language_voice_models[@]} -eq 0 ]]; then
      echo "❌ No Piper ONNX voice models found for language '${LANGUAGE}'."
      echo "   Expected files matching: piper-sample-generator/voices/${LANGUAGE}_*.onnx"
      echo "   Add voice files or choose English (en)."
      exit 1
    fi
    echo "ℹ️  No --piper-model provided; using ${#language_voice_models[@]} voice model(s) for language '${LANGUAGE}':"
    for vf in "${language_voice_models[@]}"; do
      echo "    $vf"
    done
    PIPER_MODELS=("${language_voice_models[@]}")
  fi
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

mkdir -p generated_samples

# ── (B) bulk TTS (skip if enough files present) ──────────────────────────────
count_existing=$(find generated_samples -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
if [[ "${count_existing:-0}" -lt "$MAX_TTS_SAMPLES" ]]; then
  echo "🎤 Generating ${MAX_TTS_SAMPLES} samples for '${TARGET_WORD}' (batch ${BATCH_SIZE})…"
  "$PY" - <<'PY'
import os, sys, shlex, subprocess

# make sure the generator is importable
if "piper-sample-generator/" not in sys.path:
    sys.path.append("piper-sample-generator/")

TARGET = os.environ["TARGET_WORD"]
MAX_SAMPLES = int(os.environ["MAX_TTS_SAMPLES"])
BATCH = int(os.environ["BATCH_SIZE"])
OUT_DIR = "generated_samples"

models = [m.strip() for m in os.environ.get("PIPER_MODELS_CSV","").split(",") if m.strip()]
model_flags = sum([["--model", m] for m in models], [])

# "Speed" control for Piper is length_scale; generator exposes it as --length-scales
LENGTH_SCALES = ["0.85", "0.95", "1.0", "1.05", "1.15"]

cmd = [
    sys.executable,
    "scripts_macos/run_generator_with_progress.py",
    "--generator", "piper-sample-generator/generate_samples.py",
    "--output-dir", OUT_DIR,
    "--max-samples", str(MAX_SAMPLES),
    "--",
    TARGET,
    "--max-samples", str(MAX_SAMPLES),
    "--batch-size", str(BATCH),
    "--output-dir", OUT_DIR,
    "--length-scales", *LENGTH_SCALES,
    *model_flags,
]

print("CMD:", " ".join(shlex.quote(c) for c in cmd))
proc = subprocess.run(cmd, text=True)
if proc.returncode != 0:
    raise SystemExit(proc.returncode)
PY
else
  echo "✅ Found ${count_existing} samples (>= desired); skipping TTS generation."
fi

# ── (C) pull/prepare augmentation datasets (RIR, Audioset, FMA) ──────────────
"$PY" scripts_macos/prepare_datasets.py

# ── (D) trim silence from personal samples, if any exists
"$PY" scripts_macos/trim_silence.py

# ── (E) build augmenter + spectrogram feature mmaps ───────────────────────────
"$PY" scripts_macos/make_features.py

# ── (F) download precomputed negative spectrograms ────────────────────────────
"$PY" scripts_macos/fetch_negatives.py

# ── (G) write training YAML (tuned for your notebook) ────────────────────────
"$PY" scripts_macos/write_training_yaml.py

# ── (H) train + export (Metal TF) ────────────────────────────────────────────
"$PY" -m microwakeword.model_train_eval \
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

# ── (I) package artifacts (name by wake word) ─────────────────────────────────
"$PY" - <<'PY'
import os, re, shutil, json
from pathlib import Path

target = os.environ.get("TARGET_WORD", "wakeword")
language = os.environ.get("MWW_LANGUAGE", "en")
safe = re.sub(r'[^a-z0-9_]+', '', re.sub(r'\s+', '_', target.lower()))
if not safe:
    safe = "wakeword"

src = Path("trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite")
dst = Path(f"{safe}.tflite")
if not src.exists():
    raise SystemExit(f"❌ Model not found at {src}")
shutil.copy(src, dst)

meta = {
  "type": "micro",
  "wake_word": target,
  "author": "Tater Totterson",
  "website": "https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon",
  "model": f"{safe}.tflite",
  "trained_languages": [language],
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
