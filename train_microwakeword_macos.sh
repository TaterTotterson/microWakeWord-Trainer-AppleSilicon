#!/usr/bin/env bash
# train_microwakeword_macos.sh
# One-shot: setup (idempotent) + run pipeline on Apple Silicon (macOS).
# Usage:
#   ./train_microwakeword_macos.sh "hey_tater" 50000 100 \
#       --language en --tts-mode hybrid \
#       --piper-model /path/to/voice1.onnx --piper-model /path/to/voice2.pt
#
# Hybrid uses Piper as a fourth source when a compatible model is present.

set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPORT_DIR="${WAKEWORD_TRAINER_SUPPORT_DIR:-${HOME}/.taterwakewordtrainer}"
WORK_DIR="${WAKEWORD_TRAINER_DATA_DIR:-${SUPPORT_DIR}/app/current}"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"
export WAKEWORD_TRAINER_SUPPORT_DIR="$SUPPORT_DIR"
export WAKEWORD_TRAINER_DATA_DIR="$WORK_DIR"
echo "📁 Trainer source: $SOURCE_DIR"
echo "📁 Training data: $WORK_DIR"

TARGET_WORD="${1:-hey_tater}"
MAX_TTS_SAMPLES="${2:-50000}"
BATCH_SIZE="${3:-100}"
[[ $# -ge 3 ]] && shift 3 || shift $#

# Default language can be overridden by --language or MWW_LANGUAGE
LANGUAGE="${MWW_LANGUAGE:-en}"
TTS_MODE="${MWW_TTS_MODE:-hybrid}"
TTS_VOICE_COUNT="${MWW_TTS_VOICE_COUNT:-128}"

# Collect optional flags
PIPER_MODELS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --piper-model) PIPER_MODELS+=("$2"); shift 2 ;;
    --language) LANGUAGE="${2:-}"; shift 2 ;;
    --tts-mode) TTS_MODE="${2:-}"; shift 2 ;;
    --tts-voice-count) TTS_VOICE_COUNT="${2:-}"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

LANGUAGE="$(echo "${LANGUAGE}" | tr '[:upper:]' '[:lower:]')"
if [[ -z "$LANGUAGE" ]]; then
  LANGUAGE="en"
fi
export MWW_LANGUAGE="$LANGUAGE"
echo "🌐 Training language: $LANGUAGE"

TTS_MODE="$(echo "${TTS_MODE}" | tr '[:upper:]' '[:lower:]')"
case "$TTS_MODE" in
  modern|hybrid|piper) ;;
  *) echo "❌ Invalid TTS mode '${TTS_MODE}'. Choose modern, hybrid, or piper."; exit 1 ;;
esac
if [[ ! "$TTS_VOICE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "❌ --tts-voice-count must be a positive integer."
  exit 1
fi
export MWW_TTS_MODE="$TTS_MODE" MWW_TTS_VOICE_COUNT="$TTS_VOICE_COUNT"
echo "🗣️  TTS mode: $TTS_MODE (direct final-sample generation; no reusable voice bank)"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "❌ This script is intended for macOS (Apple Silicon)."; exit 1
fi

# ── Ensure system deps ────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  echo "❌ Homebrew is required but not found. Install from https://brew.sh/ first."
  exit 1
fi

echo "📦 Ensuring FFmpeg + wget are installed (via Homebrew)…"

# wget first
brew list wget &>/dev/null || brew install wget

ffmpeg_health_check() {
  [[ -x "$FFMPEG_BIN" ]] && "$FFMPEG_BIN" \
    -nostdin \
    -hide_banner \
    -loglevel error \
    -f lavfi \
    -i "anullsrc=r=16000:cl=mono" \
    -t 0.05 \
    -ac 1 \
    -ar 16000 \
    -c:a pcm_s16le \
    -f null \
    - >/dev/null 2>&1
}

repair_selected_ffmpeg() {
  echo "🧯 $FFMPEG_FORMULA health check failed; attempting one Homebrew repair…"
  # Repair the dependency from the observed dyld failure when it is present,
  # then reinstall the selected FFmpeg formula so all links are refreshed.
  if brew list libbluray &>/dev/null; then
    brew reinstall libbluray || true
  fi
  brew reinstall "$FFMPEG_FORMULA" || true
}

select_ffmpeg_formula() {
  FFMPEG_FORMULA="$1"
  brew list "$FFMPEG_FORMULA" &>/dev/null || brew install "$FFMPEG_FORMULA"
  FFMPEG_PREFIX="$(brew --prefix "$FFMPEG_FORMULA")"
  FFMPEG_BIN="$FFMPEG_PREFIX/bin/ffmpeg"
}

# Prefer ffmpeg@7 for stable audio tooling compatibility. Keep the selected
# formula and executable paired; mixing one formula's libraries with another
# ffmpeg from PATH can leave normalization unusable after a Homebrew cleanup.
if brew info ffmpeg@7 &>/dev/null; then
  select_ffmpeg_formula "ffmpeg@7"
  if ! ffmpeg_health_check; then
    repair_selected_ffmpeg
    select_ffmpeg_formula "ffmpeg@7"
  fi
  if ! ffmpeg_health_check; then
    echo "⚠️ ffmpeg@7 is still unhealthy; trying the default FFmpeg formula."
    select_ffmpeg_formula "ffmpeg"
  fi
else
  echo "⚠️ ffmpeg@7 is unavailable; using the default FFmpeg formula."
  select_ffmpeg_formula "ffmpeg"
fi

if ! ffmpeg_health_check; then
  repair_selected_ffmpeg
  select_ffmpeg_formula "$FFMPEG_FORMULA"
fi
if ! ffmpeg_health_check; then
  echo "❌ FFmpeg is installed but cannot perform a basic audio conversion."
  echo "   Selected executable: $FFMPEG_BIN"
  echo "   Repair manually with: brew reinstall libbluray $FFMPEG_FORMULA"
  "$FFMPEG_BIN" -version || true
  exit 1
fi

export FFMPEG_BIN
export MWW_FFMPEG_BIN="$FFMPEG_BIN"
export PATH="$FFMPEG_PREFIX/bin:$PATH"
FFMPEG_VERSION_OUTPUT="$("$FFMPEG_BIN" -version 2>/dev/null)"
echo "✅ Using $(printf '%s\n' "$FFMPEG_VERSION_OUTPUT" | sed -n '1p')"

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

# The app launcher prepares REC_VENV_DIR before it starts the UI backend. Keep
# that working environment entirely app-managed. Standalone/headless CLI runs
# do not have it, so give them a small isolated QA environment of their own.
REFERENCE_QA_DEPENDENCIES=(
  "faster-whisper>=1.0.0"
  "silero-vad>=5.0.0"
  "numpy>=1.24.0"
)
CLI_MANAGES_REFERENCE_QA=0

reference_qa_dependencies_ready() {
  local qa_python="$1"
  "$qa_python" - <<'PY' >/dev/null 2>&1
import ctranslate2
import numpy
import onnxruntime
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
PY
}

ensure_reference_qa_environment() {
  local qa_venv qa_python pin_file expected_fingerprint installed_fingerprint

  if [[ -n "${REC_VENV_DIR:-}" ]]; then
    qa_venv="$REC_VENV_DIR"
    qa_python="$qa_venv/bin/python"
    if [[ ! -x "$qa_python" ]] || ! reference_qa_dependencies_ready "$qa_python"; then
      echo "❌ The app-managed reference QA environment is unavailable: $qa_venv"
      echo "   Restart the WakeWord Trainer app so it can repair its recorder dependencies."
      return 1
    fi
    echo "✅ Using app-managed reference QA environment: $qa_venv"
    return 0
  fi

  CLI_MANAGES_REFERENCE_QA=1
  qa_venv="${MWW_CLI_QA_VENV_DIR:-${SUPPORT_DIR}/cli-reference-qa-venv}"
  qa_python="$qa_venv/bin/python"
  pin_file="$qa_venv/.dependency_fingerprint"
  expected_fingerprint="$(printf '%s\n' "${REFERENCE_QA_DEPENDENCIES[@]}" | /usr/bin/shasum -a 256 | awk '{print $1}')"

  if [[ ! -x "$qa_python" ]]; then
    echo "🧪 Creating standalone CLI reference QA environment: $qa_venv"
    arch -arm64 "$PYTHON_BIN" -m venv "$qa_venv"
  fi

  installed_fingerprint="$(tr -d '[:space:]' < "$pin_file" 2>/dev/null || true)"
  if [[ "$installed_fingerprint" != "$expected_fingerprint" ]] || ! reference_qa_dependencies_ready "$qa_python"; then
    echo "📦 Installing standalone CLI reference QA dependencies…"
    "$qa_python" -m pip install -U pip setuptools wheel
    "$qa_python" -m pip install "${REFERENCE_QA_DEPENDENCIES[@]}"
    printf '%s\n' "$expected_fingerprint" > "$pin_file"
  fi

  if ! reference_qa_dependencies_ready "$qa_python"; then
    echo "❌ Standalone CLI reference QA environment is incomplete: $qa_venv"
    return 1
  fi

  export REC_VENV_DIR="$qa_venv"
  echo "✅ Standalone CLI reference QA environment ready: $qa_venv"
}

configure_cli_omnivoice_tmpdir() {
  local fallback_temp socket_probe
  if [[ "$CLI_MANAGES_REFERENCE_QA" != "1" ]]; then
    return 0
  fi
  fallback_temp="/tmp/tw-omni-$(id -u)"
  if [[ -z "${MWW_OMNIVOICE_TMPDIR:-}" ]]; then
    export MWW_OMNIVOICE_TMPDIR="$fallback_temp"
  else
    socket_probe="${MWW_OMNIVOICE_TMPDIR%/}/pymp-12345678/listener-1234567890abcdef"
    if (( ${#socket_probe} >= 104 )); then
      echo "⚠️ OmniVoice socket temp path is too long; using $fallback_temp"
      export MWW_OMNIVOICE_TMPDIR="$fallback_temp"
    fi
  fi
  mkdir -p "$MWW_OMNIVOICE_TMPDIR"
  chmod 700 "$MWW_OMNIVOICE_TMPDIR" 2>/dev/null || true
  echo "✅ OmniVoice CLI socket temp: $MWW_OMNIVOICE_TMPDIR"
}

TF_VERSION="${TF_VERSION:-2.16.2}"
TF_METAL_VERSION="${TF_METAL_VERSION:-1.2.0}"
KERAS_VERSION="${KERAS_VERSION:-3.3.3}"
PROTOBUF_VERSION="${PROTOBUF_VERSION:-4.25.8}"
FLATBUFFERS_VERSION="${FLATBUFFERS_VERSION:-23.5.26}"
TORCH_VERSION="${TORCH_VERSION:-2.9.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-${TORCH_VERSION}}"

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
export TORCH_VERSION TORCHAUDIO_VERSION

ensure_torch_audio_stack() {
  if "$PY" - <<PY
import importlib.metadata as md
import sys

expected = {
    "torch": "${TORCH_VERSION}",
    "torchaudio": "${TORCHAUDIO_VERSION}",
}
bad = []
for package, expected_version in expected.items():
    try:
        actual = md.version(package).split("+", 1)[0]
    except md.PackageNotFoundError:
        actual = "<missing>"
    if actual != expected_version:
        bad.append((package, actual, expected_version))

if bad:
    print("Torch audio stack drift detected:")
    for package, actual, expected_version in bad:
        print(f"  - {package}: {actual} (expected {expected_version})")
    sys.exit(1)
PY
  then
    echo "✅ Torch audio stack verified."
  else
    echo "🧯 Repairing torch audio stack to torch==${TORCH_VERSION}, torchaudio==${TORCHAUDIO_VERSION}…"
    "$PY" -m pip install -q "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"
  fi
}

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
  "$PY" -m pip install "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}"

  touch ".venv/.pinned_installed"
else
  echo "✅ Reusing existing .venv (no upgrades)"
fi

ensure_torch_audio_stack

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
import importlib.metadata as md
import tensorflow as tf
import keras
import google.protobuf
import flatbuffers
import torch

expected = {
  "tensorflow": "${TF_VERSION}",
  "keras": "${KERAS_VERSION}",
  "protobuf": "${PROTOBUF_VERSION}",
  "flatbuffers": "${FLATBUFFERS_VERSION}",
  "torch": "${TORCH_VERSION}",
  "torchaudio": "${TORCHAUDIO_VERSION}",
}

actual = {
  "tensorflow": tf.__version__,
  "keras": keras.__version__,
  "protobuf": google.protobuf.__version__,
  "flatbuffers": flatbuffers.__version__,
  "torch": torch.__version__.split("+", 1)[0],
  "torchaudio": md.version("torchaudio").split("+", 1)[0],
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

# Piper is no longer installed for the default modern stack. It remains
# available as an explicit compatibility/fallback option.
if [[ "$TTS_MODE" == "piper" ]]; then
  bash "$SOURCE_DIR/scripts_macos/get_piper_generator.sh"
  ensure_torch_audio_stack
elif [[ "$TTS_MODE" == "hybrid" ]]; then
  if bash "$SOURCE_DIR/scripts_macos/get_piper_generator.sh"; then
    ensure_torch_audio_stack
  else
    echo "⚠️  Piper setup failed; hybrid mode will continue with modern TTS only."
  fi
fi

# ── verify Metal GPU (optional) ───────────────────────────────────────────────
"$PY" - <<'PY'
import tensorflow as tf
devs = tf.config.list_logical_devices()
print("✅ TF logical devices:", [d.name for d in devs])
if not any(d.device_type == "GPU" for d in devs):
    print("⚠️  No GPU logical device detected. Will run on CPU.")
PY

# ── export for inline python ──────────────────────────────────────────────────
export TARGET_WORD MAX_TTS_SAMPLES BATCH_SIZE LANGUAGE MWW_LANGUAGE TTS_MODE TTS_VOICE_COUNT

# ── Resolve optional Piper compatibility voices ──────────────────────────────
DEFAULT_MODEL_PT="piper-sample-generator/models/en_US-libritts_r-medium.pt"
if [[ "$TTS_MODE" == "modern" && ${#PIPER_MODELS[@]} -gt 0 ]]; then
  echo "⚠️  --piper-model is ignored in modern mode. Use --tts-mode hybrid or piper."
  PIPER_MODELS=()
elif [[ "$TTS_MODE" != "modern" && ${#PIPER_MODELS[@]} -eq 0 ]]; then
  if [[ "$LANGUAGE" == "en" ]]; then
    echo "ℹ️  No --piper-model provided; using default English voice:"
    echo "    $DEFAULT_MODEL_PT"
    mkdir -p "$(dirname "$DEFAULT_MODEL_PT")"
    if [[ ! -f "$DEFAULT_MODEL_PT" ]]; then
      if ! wget -q -O "$DEFAULT_MODEL_PT" \
        "https://github.com/TaterTotterson/piper-sample-generator/releases/download/models/en_US-libritts_r-medium.pt"; then
        rm -f "$DEFAULT_MODEL_PT"
        if [[ "$TTS_MODE" == "piper" ]]; then
          echo "❌ Could not download the required English Piper voice."
          exit 1
        fi
        echo "⚠️  Could not download the English Piper voice; hybrid mode will use modern TTS only."
      fi
    fi
    if [[ -f "$DEFAULT_MODEL_PT" ]]; then
      PIPER_MODELS=("$DEFAULT_MODEL_PT")
    fi
  else
    shopt -s nullglob
    language_voice_models=(piper-sample-generator/voices/"${LANGUAGE}"_*.onnx)
    shopt -u nullglob
    if [[ ${#language_voice_models[@]} -eq 0 ]]; then
      if [[ "$TTS_MODE" == "piper" ]]; then
        echo "❌ No Piper ONNX voice models found for language '${LANGUAGE}'."
        echo "   Expected files matching: piper-sample-generator/voices/${LANGUAGE}_*.onnx"
        exit 1
      fi
      echo "⚠️  No Piper voice found for '${LANGUAGE}'; hybrid mode will use modern engines only."
    else
      echo "ℹ️  Using ${#language_voice_models[@]} Piper compatibility voice(s) for '${LANGUAGE}':"
      for vf in "${language_voice_models[@]}"; do
        echo "    $vf"
      done
      PIPER_MODELS=("${language_voice_models[@]}")
    fi
  fi
fi

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
    printf 'tts_mode=%s\n' "$TTS_MODE"
    for generator_file in \
      "$SOURCE_DIR/tts_config.py" \
      "$SOURCE_DIR/scripts_macos/tts_generate_samples.py" \
      "$SOURCE_DIR/scripts_macos/tts_qwen_mlx_worker.py" \
      "$SOURCE_DIR/scripts_macos/tts_moss_mlx_worker.py" \
      "$SOURCE_DIR/scripts_macos/setup_modern_tts_envs"; do
      stat -f 'generator=%N:%m:%z' "$generator_file"
    done
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

compute_reviewed_negative_cache_key() {
  if ! dir_has_matching_files "negative_samples" "*.wav"; then
    echo "none"
    return
  fi
  {
    find "negative_samples" -type f -name '*.wav' -exec stat -f 'negative=%N:%m:%z' {} \; | sort
  } | shasum -a 256 | awk '{print $1}'
}

compute_feature_cache_key() {
  local sample_key="$1"
  local personal_key="$2"
  local reviewed_negative_key="$3"
  {
    printf 'sample_key=%s\n' "$sample_key"
    printf 'personal_key=%s\n' "$personal_key"
    printf 'reviewed_negative_key=%s\n' "$reviewed_negative_key"
    stat -f 'feature_script=%N:%m:%z' "$SOURCE_DIR/scripts_macos/make_features.py"
    for dataset_dir in mit_rirs audioset_16k fma_16k wham_16k chime_16k; do
      printf '%s=%s\n' "$dataset_dir" "$(count_matching_files "$dataset_dir" '*.wav')"
    done
  } | shasum -a 256 | awk '{print $1}'
}

SAMPLE_CACHE_KEY_FILE="generated_samples/.cache_key"
SAMPLE_CACHE_STAMP_FILE="generated_samples/.cache_stamp"
FEATURE_CACHE_KEY_FILE="generated_augmented_features/.cache_key"
PERSONAL_FEATURE_CACHE_KEY_FILE="personal_augmented_features/.cache_key"
REVIEWED_NEGATIVE_FEATURE_CACHE_KEY_FILE="reviewed_negative_features/.cache_key"
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
  # Do this before launching any expensive TTS engine. A missing QA runtime
  # must fail immediately rather than after hours of successful synthesis.
  ensure_reference_qa_environment
  configure_cli_omnivoice_tmpdir
  echo "🎤 Generating ${MAX_TTS_SAMPLES} samples for '${TARGET_WORD}' with ${TTS_MODE} TTS…"
  generator_cmd=(
    "$PY"
    "$SOURCE_DIR/scripts_macos/tts_generate_samples.py"
    "$TARGET_WORD"
    "--language" "$LANGUAGE"
    "--tts-mode" "$TTS_MODE"
    "--samples" "$MAX_TTS_SAMPLES"
    "--batch-size" "$BATCH_SIZE"
    "--voice-count" "$TTS_VOICE_COUNT"
    "--data-dir" "$WORK_DIR"
    "--output-dir" "generated_samples"
    "--ffmpeg" "$FFMPEG_BIN"
  )
  for model_path in "${PIPER_MODELS[@]}"; do
    generator_cmd+=("--piper-model" "$model_path")
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
"$PY" "$SOURCE_DIR/scripts_macos/prepare_datasets.py"

# ── (D) trim silence from personal samples, if any exists
if dir_has_matching_files "personal_samples" "*.wav"; then
  echo "✂️ Trimming silence from personal samples…"
  "$PY" "$SOURCE_DIR/scripts_macos/trim_silence.py"
else
  echo "ℹ️ No personal samples uploaded; skipping silence trimming."
fi

# ── (E) build augmenter + spectrogram feature mmaps ───────────────────────────
PERSONAL_CACHE_KEY="$(compute_personal_cache_key)"
REVIEWED_NEGATIVE_CACHE_KEY="$(compute_reviewed_negative_cache_key)"
SAMPLE_CACHE_STAMP="$(read_cache_key "$SAMPLE_CACHE_STAMP_FILE")"
FEATURE_CACHE_KEY="$(compute_feature_cache_key "${SAMPLE_CACHE_KEY}:${SAMPLE_CACHE_STAMP}" "$PERSONAL_CACHE_KEY" "$REVIEWED_NEGATIVE_CACHE_KEY")"
feature_cache_hit=false
cached_feature_key="$(read_cache_key "$FEATURE_CACHE_KEY_FILE")"
cached_personal_feature_key="$(read_cache_key "$PERSONAL_FEATURE_CACHE_KEY_FILE")"
cached_reviewed_negative_feature_key="$(read_cache_key "$REVIEWED_NEGATIVE_FEATURE_CACHE_KEY_FILE")"

if features_dir_ready "generated_augmented_features" && [[ -n "$cached_feature_key" && "$cached_feature_key" == "$FEATURE_CACHE_KEY" ]]; then
  personal_feature_cache_ok=false
  if [[ "$PERSONAL_CACHE_KEY" == "none" ]]; then
    personal_feature_cache_ok=true
    if [[ -d "personal_augmented_features" ]]; then
      echo "♻️ Removing stale personal feature cache (no personal samples present)."
      rm -rf personal_augmented_features
    fi
  elif features_dir_ready "personal_augmented_features" && [[ -n "$cached_personal_feature_key" && "$cached_personal_feature_key" == "$FEATURE_CACHE_KEY" ]]; then
    personal_feature_cache_ok=true
  fi

  reviewed_negative_feature_cache_ok=false
  if [[ "$REVIEWED_NEGATIVE_CACHE_KEY" == "none" ]]; then
    reviewed_negative_feature_cache_ok=true
    if [[ -d "reviewed_negative_features" ]]; then
      echo "♻️ Removing stale reviewed negative feature cache (no reviewed negative samples present)."
      rm -rf reviewed_negative_features
    fi
  elif features_dir_ready "reviewed_negative_features" && [[ -n "$cached_reviewed_negative_feature_key" && "$cached_reviewed_negative_feature_key" == "$FEATURE_CACHE_KEY" ]]; then
    reviewed_negative_feature_cache_ok=true
  fi

  if [[ "$personal_feature_cache_ok" == "true" && "$reviewed_negative_feature_cache_ok" == "true" ]]; then
    feature_cache_hit=true
  fi
fi

if [[ "$feature_cache_hit" == "true" ]]; then
  echo "✅ Reusing augmented feature caches for the current wake word, personal samples, and reviewed negatives."
else
  if [[ -d "generated_augmented_features" || -d "personal_augmented_features" || -d "reviewed_negative_features" ]]; then
    echo "♻️ Feature cache changed; rebuilding augmented features."
    rm -rf generated_augmented_features personal_augmented_features reviewed_negative_features
  fi
  echo "🧪 Building augmented feature sets…"
  "$PY" "$SOURCE_DIR/scripts_macos/make_features.py"
  write_cache_key "$FEATURE_CACHE_KEY_FILE" "$FEATURE_CACHE_KEY"
  if [[ "$PERSONAL_CACHE_KEY" != "none" && -d "personal_augmented_features" ]]; then
    write_cache_key "$PERSONAL_FEATURE_CACHE_KEY_FILE" "$FEATURE_CACHE_KEY"
  fi
  if [[ "$REVIEWED_NEGATIVE_CACHE_KEY" != "none" && -d "reviewed_negative_features" ]]; then
    write_cache_key "$REVIEWED_NEGATIVE_FEATURE_CACHE_KEY_FILE" "$FEATURE_CACHE_KEY"
  fi
fi

# ── (F) download precomputed negative spectrograms ────────────────────────────
echo "⬇️ Fetching negative datasets…"
"$PY" "$SOURCE_DIR/scripts_macos/fetch_negatives.py"

# ── (G) write training YAML (tuned for your notebook) ────────────────────────
echo "📝 Writing training config…"
"$PY" "$SOURCE_DIR/scripts_macos/write_training_yaml.py"

# ── (H) train + export (Metal TF) ────────────────────────────────────────────
echo "🏋️ Starting model training and TFLite export (this is the longest stage)…"
echo "🧠 Model quality: high_accuracy_plus"
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
  --pointwise_filters "128,128,128,128" \
  --repeat_in_block "1,1,1,1" \
  --mixconv_kernel_sizes "[5], [7,11], [9,15], [23]" \
  --residual_connection "0,0,0,0" \
  --first_conv_filters 64 \
  --first_conv_kernel_size 5 \
  --stride 2

# ── (I) calibrate detector metadata ────────────────────────────────────────────
CALIBRATION_JSON="trained_models/wakeword/tflite_stream_state_internal_quant/detection_calibration.json"
echo "🎯 Calibrating detector settings for on-device use…"
if "$PY" "$SOURCE_DIR/scripts_macos/calibrate_detector.py" \
  --training-config "trained_models/wakeword/training_config.yaml" \
  --model "trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite" \
  --output "$CALIBRATION_JSON" \
  --target-faph "${MWW_CALIBRATION_TARGET_FAPH:-0.25}" \
  --recall-margin "${MWW_CALIBRATION_RECALL_MARGIN:-0.005}" \
  --window-sizes "${MWW_CALIBRATION_WINDOW_SIZES:-5,6,7}" \
  --cutoff-min "${MWW_CALIBRATION_CUTOFF_MIN:-0.95}" \
  --cutoff-max "${MWW_CALIBRATION_CUTOFF_MAX:-1.00}"; then
  echo "✅ Detector calibration complete."
else
  echo "⚠️ Detector calibration failed; packaging with default detector settings."
  rm -f "$CALIBRATION_JSON"
fi

# ── (J) package artifacts (name by wake word) ─────────────────────────────────
echo "📦 Packaging final model artifacts…"
"$PY" "$SOURCE_DIR/scripts_macos/package_model.py" \
  "$TARGET_WORD" \
  "$LANGUAGE" \
  "$CALIBRATION_JSON" \
  --output-dir "${TRAINED_WAKE_WORDS_DIR:-trained_wake_words}" \
  --name-by-wake-word

echo "🎉 Done."
