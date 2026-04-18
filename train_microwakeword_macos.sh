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

# prefer ffmpeg@7 for stable audio tooling compatibility
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

# Make the chosen ffmpeg visible to local audio tooling on macOS (ARM sometimes needs DYLD_*)
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
export PYTHONUNBUFFERED=1

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

  # Pinned torch stack
  "$PY" -m pip install "torch==${TORCH_VERSION}"

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

count_matching_files() {
  local dir="$1"
  local pattern="$2"
  if [[ -d "$dir" ]]; then
    find "$dir" -type f -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
  else
    echo "0"
  fi
}

dir_has_matching_files() {
  local dir="$1"
  local pattern="$2"
  local first_match=""
  if [[ -d "$dir" ]]; then
    first_match=$(find "$dir" -type f -name "$pattern" -print -quit 2>/dev/null || true)
  fi
  [[ -n "$first_match" ]]
}

features_dir_ready() {
  local dir="$1"
  [[ -d "$dir/training" && -d "$dir/validation" && -d "$dir/testing" ]]
}

read_cache_key() {
  local key_file="$1"
  if [[ -f "$key_file" ]]; then
    tr -d '\n' < "$key_file"
  fi
}

write_cache_key() {
  local key_file="$1"
  local key_value="$2"
  mkdir -p "$(dirname "$key_file")"
  printf '%s\n' "$key_value" > "$key_file"
}

compute_sample_cache_key() {
  {
    printf 'target=%s\n' "$TARGET_WORD"
    printf 'samples=%s\n' "$MAX_TTS_SAMPLES"
    printf 'batch=%s\n' "$BATCH_SIZE"
    printf 'language=%s\n' "$LANGUAGE"
    stat -f 'generator_wrapper=%N:%m:%z' "scripts_macos/run_generator_with_progress.py"
    for model_path in "${PIPER_MODELS[@]}"; do
      if [[ -e "$model_path" ]]; then
        stat -f 'model=%N:%m:%z' "$model_path"
      else
        printf 'model_missing=%s\n' "$model_path"
      fi
    done
  } | shasum -a 256 | awk '{print $1}'
}

compute_personal_cache_key() {
  if ! dir_has_matching_files "personal_samples" "*.wav"; then
    echo "none"
    return
  fi
  {
    find "personal_samples" -type f -name '*.wav' -exec stat -f 'personal=%N:%m:%z' {} \; | sort
  } | shasum -a 256 | awk '{print $1}'
}

compute_feature_cache_key() {
  local sample_key="$1"
  local personal_key="$2"
  {
    printf 'sample_key=%s\n' "$sample_key"
    printf 'personal_key=%s\n' "$personal_key"
    stat -f 'feature_script=%N:%m:%z' "scripts_macos/make_features.py"
    for dataset_dir in mit_rirs audioset_16k fma_16k wham_16k chime_16k; do
      printf '%s=%s\n' "$dataset_dir" "$(count_matching_files "$dataset_dir" '*.wav')"
    done
  } | shasum -a 256 | awk '{print $1}'
}

SAMPLE_CACHE_KEY_FILE="generated_samples/.cache_key"
SAMPLE_CACHE_STAMP_FILE="generated_samples/.cache_stamp"
FEATURE_CACHE_KEY_FILE="generated_augmented_features/.cache_key"
PERSONAL_FEATURE_CACHE_KEY_FILE="personal_augmented_features/.cache_key"
SAMPLE_CACHE_KEY="$(compute_sample_cache_key)"

# ── (A) clean previous run artifacts that must always be rebuilt ─────────────
echo "🧹 Cleaning previous training outputs…"
rm -f training_parameters.yaml
rm -rf trained_models
echo "✅ Training outputs cleared."

mkdir -p generated_samples

# ── (B) bulk TTS (skip if enough files present) ──────────────────────────────
sample_cache_hit=false
count_existing=$(count_matching_files "generated_samples" "*.wav")
cached_sample_key="$(read_cache_key "$SAMPLE_CACHE_KEY_FILE")"
cached_sample_stamp="$(read_cache_key "$SAMPLE_CACHE_STAMP_FILE")"
if [[ "${count_existing:-0}" -eq "$MAX_TTS_SAMPLES" && -n "$cached_sample_key" && -n "$cached_sample_stamp" && "$cached_sample_key" == "$SAMPLE_CACHE_KEY" ]]; then
  sample_cache_hit=true
  echo "✅ Reusing generated samples for the same wake word and voice setup."
else
  if [[ "${count_existing:-0}" -gt 0 || -n "$cached_sample_key" || -n "$cached_sample_stamp" ]]; then
    echo "♻️ Generated sample cache changed or is incomplete; rebuilding generated samples."
    rm -rf generated_samples
    mkdir -p generated_samples
  fi
fi

if [[ "$sample_cache_hit" != "true" ]]; then
  echo "🎤 Generating ${MAX_TTS_SAMPLES} samples for '${TARGET_WORD}' (batch ${BATCH_SIZE})…"
  LENGTH_SCALES=(0.85 0.95 1.0 1.05 1.15)
  generator_cmd=(
    "$PY"
    "scripts_macos/run_generator_with_progress.py"
    "--generator" "piper-sample-generator/generate_samples.py"
    "--output-dir" "generated_samples"
    "--max-samples" "$MAX_TTS_SAMPLES"
    "--"
    "$TARGET_WORD"
    "--max-samples" "$MAX_TTS_SAMPLES"
    "--batch-size" "$BATCH_SIZE"
    "--output-dir" "generated_samples"
    "--length-scales"
  )

  for scale in "${LENGTH_SCALES[@]}"; do
    generator_cmd+=("$scale")
  done

  for model_path in "${PIPER_MODELS[@]}"; do
    generator_cmd+=("--model" "$model_path")
  done

  printf 'CMD:'
  printf ' %q' "${generator_cmd[@]}"
  printf '\n'
  "${generator_cmd[@]}"
  generated_files=$(count_matching_files "generated_samples" "*.wav")
  if [[ "${generated_files:-0}" -ne "$MAX_TTS_SAMPLES" ]]; then
    echo "❌ Expected ${MAX_TTS_SAMPLES} generated samples, but found ${generated_files}."
    exit 1
  fi
  write_cache_key "$SAMPLE_CACHE_KEY_FILE" "$SAMPLE_CACHE_KEY"
  write_cache_key "$SAMPLE_CACHE_STAMP_FILE" "${SAMPLE_CACHE_KEY}:$(date +%s)"
else
  echo "ℹ️ Skipping TTS generation because cached samples are still valid."
fi

# ── (C) pull/prepare augmentation datasets (RIR, Audioset, FMA) ──────────────
echo "📚 Preparing augmentation datasets (MIT RIR, AudioSet, FMA, WHAM, CHiME)…"
"$PY" scripts_macos/prepare_datasets.py

# ── (D) trim silence from personal samples, if any exists
if dir_has_matching_files "personal_samples" "*.wav"; then
  echo "✂️ Trimming silence from personal samples…"
  "$PY" scripts_macos/trim_silence.py
else
  echo "ℹ️ No personal samples uploaded; skipping silence trimming."
fi

# ── (E) build augmenter + spectrogram feature mmaps ───────────────────────────
PERSONAL_CACHE_KEY="$(compute_personal_cache_key)"
SAMPLE_CACHE_STAMP="$(read_cache_key "$SAMPLE_CACHE_STAMP_FILE")"
FEATURE_CACHE_KEY="$(compute_feature_cache_key "${SAMPLE_CACHE_KEY}:${SAMPLE_CACHE_STAMP}" "$PERSONAL_CACHE_KEY")"
feature_cache_hit=false
cached_feature_key="$(read_cache_key "$FEATURE_CACHE_KEY_FILE")"
cached_personal_feature_key="$(read_cache_key "$PERSONAL_FEATURE_CACHE_KEY_FILE")"

if features_dir_ready "generated_augmented_features" && [[ -n "$cached_feature_key" && "$cached_feature_key" == "$FEATURE_CACHE_KEY" ]]; then
  if [[ "$PERSONAL_CACHE_KEY" == "none" ]]; then
    feature_cache_hit=true
    if [[ -d "personal_augmented_features" ]]; then
      echo "♻️ Removing stale personal feature cache (no personal samples present)."
      rm -rf personal_augmented_features
    fi
  elif features_dir_ready "personal_augmented_features" && [[ -n "$cached_personal_feature_key" && "$cached_personal_feature_key" == "$FEATURE_CACHE_KEY" ]]; then
    feature_cache_hit=true
  fi
fi

if [[ "$feature_cache_hit" == "true" ]]; then
  echo "✅ Reusing augmented feature caches for the current wake word and personal samples."
else
  if [[ -d "generated_augmented_features" || -d "personal_augmented_features" ]]; then
    echo "♻️ Feature cache changed; rebuilding augmented features."
    rm -rf generated_augmented_features personal_augmented_features
  fi
  echo "🧪 Building augmented feature sets…"
  "$PY" scripts_macos/make_features.py
  write_cache_key "$FEATURE_CACHE_KEY_FILE" "$FEATURE_CACHE_KEY"
  if [[ "$PERSONAL_CACHE_KEY" != "none" && -d "personal_augmented_features" ]]; then
    write_cache_key "$PERSONAL_FEATURE_CACHE_KEY_FILE" "$FEATURE_CACHE_KEY"
  fi
fi

# ── (F) download precomputed negative spectrograms ────────────────────────────
echo "⬇️ Fetching negative datasets…"
"$PY" scripts_macos/fetch_negatives.py

# ── (G) write training YAML (tuned for your notebook) ────────────────────────
echo "📝 Writing training config…"
"$PY" scripts_macos/write_training_yaml.py

# ── (H) train + export (Metal TF) ────────────────────────────────────────────
echo "🏋️ Starting model training and TFLite export (this is the longest stage)…"
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

# ── (I) calibrate detector metadata ────────────────────────────────────────────
CALIBRATION_JSON="trained_models/wakeword/tflite_stream_state_internal_quant/detection_calibration.json"
echo "🎯 Calibrating detector settings for on-device use…"
if "$PY" scripts_macos/calibrate_detector.py \
  --training-config "trained_models/wakeword/training_config.yaml" \
  --model "trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite" \
  --output "$CALIBRATION_JSON"; then
  echo "✅ Detector calibration complete."
else
  echo "⚠️ Detector calibration failed; packaging with default detector settings."
  rm -f "$CALIBRATION_JSON"
fi

# ── (J) package artifacts (name by wake word) ─────────────────────────────────
echo "📦 Packaging final model artifacts…"
"$PY" - <<'PY'
import os, re, shutil, json
from pathlib import Path

target = os.environ.get("TARGET_WORD", "wakeword")
language = os.environ.get("MWW_LANGUAGE", "en")
calibration_path = Path(
    "trained_models/wakeword/tflite_stream_state_internal_quant/detection_calibration.json"
)
safe = re.sub(r'[^a-z0-9_]+', '', re.sub(r'\s+', '_', target.lower()))
if not safe:
    safe = "wakeword"

src = Path("trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite")
dst = Path(f"{safe}.tflite")
if not src.exists():
    raise SystemExit(f"❌ Model not found at {src}")
shutil.copy(src, dst)

probability_cutoff = 0.97
sliding_window_size = 5
if calibration_path.exists():
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        probability_cutoff = float(calibration.get("probability_cutoff", probability_cutoff))
        sliding_window_size = int(calibration.get("sliding_window_size", sliding_window_size))
        print(
            f"🎯 Using calibrated detector settings: "
            f"cutoff={probability_cutoff:.2f}, window={sliding_window_size}"
        )
    except Exception as exc:
        print(f"⚠️ Failed to read detector calibration ({exc}); using defaults.")

meta = {
  "type": "micro",
  "wake_word": target,
  "author": "Tater Totterson",
  "website": "https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon",
  "model": f"{safe}.tflite",
  "trained_languages": [language],
  "version": 2,
  "micro": {
    "probability_cutoff": round(probability_cutoff, 2),
    "sliding_window_size": sliding_window_size,
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
