#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd -P)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd -P)"
APP_NAME="WakeWord Trainer"
APP_DIR="${PROJECT_DIR}/build/${APP_NAME}.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
SOURCE_SNAPSHOT_DIR="${RESOURCES_DIR}/TrainerSource"
CODESIGN_IDENTITY="${WAKEWORD_TRAINER_CODESIGN_IDENTITY:--}"

swift build -c release --package-path "${PROJECT_DIR}"
BIN_DIR="$(swift build -c release --package-path "${PROJECT_DIR}" --show-bin-path)"

"${SCRIPT_DIR}/generate_app_icon.sh"

rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}" "${RESOURCES_DIR}"

cp "${BIN_DIR}/WakeWordTrainer" "${MACOS_DIR}/WakeWordTrainer"
cp "${PROJECT_DIR}/Resources/Info.plist" "${CONTENTS_DIR}/Info.plist"
cp "${PROJECT_DIR}/Resources/WakeWordIcon.icns" "${RESOURCES_DIR}/WakeWordIcon.icns"
cp "${PROJECT_DIR}/Resources/WakeWordIconSource.png" "${RESOURCES_DIR}/WakeWordIconSource.png"
cp "${REPO_ROOT}/images/tater-repo-logo.png" "${RESOURCES_DIR}/TaterRepoLogo.png"
cp "${PROJECT_DIR}/Resources/WakeWordMenuBarTemplate.png" "${RESOURCES_DIR}/WakeWordMenuBarTemplate.png"
rsync -a --delete \
  --exclude='.git/' \
  --exclude='.github/' \
  --exclude='.agents/' \
  --exclude='.codex/' \
  --exclude='.recorder-venv/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  --exclude='macos/' \
  --exclude='personal_samples/' \
  --exclude='negative_samples/' \
  --exclude='captured_audio/' \
  --exclude='trim_history/' \
  --exclude='trained_wake_words/' \
  --exclude='trained_models/' \
  --exclude='generated_samples/' \
  --exclude='generated_augmented_features/' \
  --exclude='personal_augmented_features/' \
  --exclude='reviewed_negative_features/' \
  --exclude='micro-wake-word/' \
  --exclude='piper-sample-generator/' \
  --exclude='mit_rirs/' \
  --exclude='audioset/' \
  --exclude='audioset_16k/' \
  --exclude='fma/' \
  --exclude='fma_16k/' \
  --exclude='wham_16k/' \
  --exclude='chime_16k/' \
  "${REPO_ROOT}/" "${SOURCE_SNAPSHOT_DIR}/"

chmod +x "${MACOS_DIR}/WakeWordTrainer"
chmod +x "${SOURCE_SNAPSHOT_DIR}/run.sh" "${SOURCE_SNAPSHOT_DIR}/train_microwakeword_macos.sh" 2>/dev/null || true

find "${APP_DIR}" -exec xattr -c {} +
codesign --force --deep --sign "${CODESIGN_IDENTITY}" "${APP_DIR}"
codesign --verify --deep --strict --verbose=2 "${APP_DIR}"

printf 'Built %s\n' "${APP_DIR}"
