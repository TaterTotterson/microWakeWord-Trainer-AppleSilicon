#!/bin/sh
set -eu
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd -P)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
APP_DIR="${PROJECT_DIR}/build/WakeWord Trainer.app"
INFO_PLIST="${PROJECT_DIR}/Resources/Info.plist"
RELEASES_DIR="${PROJECT_DIR}/releases"

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "${INFO_PLIST}")"
BUILD="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "${INFO_PLIST}")"
VERSION_TOKEN="$(printf '%s' "${VERSION}" | sed 's/^[vV]//')"
VERSION_LABEL="v${VERSION_TOKEN}"
BUILD_NUMBER="$(printf '%s' "${BUILD}" | sed 's/[^0-9].*$//')"
if [ -z "${BUILD_NUMBER}" ]; then
  BUILD_NUMBER="$(printf '%s' "${VERSION_TOKEN}" | sed 's/[^0-9].*$//')"
fi
if [ -z "${BUILD_NUMBER}" ]; then
  BUILD_NUMBER="0"
fi
ZIP_NAME="WakeWordTrainer-${VERSION_LABEL}.zip"
ZIP_PATH="${PROJECT_DIR}/build/${ZIP_NAME}"
RELEASE_ZIP_PATH="${RELEASES_DIR}/${ZIP_NAME}"
MANIFEST_PATH="${PROJECT_DIR}/build/update-manifest.json"
REPO_MANIFEST_PATH="${PROJECT_DIR}/update-manifest.json"
DOWNLOAD_URL="${1:-https://raw.githubusercontent.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon/main/macos/WakeWordTrainer/releases/${ZIP_NAME}}"

"${SCRIPT_DIR}/build_app.sh"

mkdir -p "${RELEASES_DIR}"
rm -f "${ZIP_PATH}" "${MANIFEST_PATH}" "${RELEASE_ZIP_PATH}"
ditto -c -k --keepParent "${APP_DIR}" "${ZIP_PATH}"
sh "${SCRIPT_DIR}/notarize_artifact.sh" "${ZIP_PATH}"
if [ "${WAKEWORD_TRAINER_NOTARIZE:-0}" = "1" ]; then
  xcrun stapler staple "${APP_DIR}"
  xcrun stapler validate "${APP_DIR}"
  rm -f "${ZIP_PATH}"
  ditto -c -k --keepParent "${APP_DIR}" "${ZIP_PATH}"
fi
cp "${ZIP_PATH}" "${RELEASE_ZIP_PATH}"
SHA256="$(shasum -a 256 "${ZIP_PATH}" | awk '{print $1}')"

{
  printf '{\n'
  printf '  "version": "%s",\n' "${VERSION_TOKEN}"
  printf '  "build": %s,\n' "${BUILD_NUMBER}"
  printf '  "url": "%s",\n' "${DOWNLOAD_URL}"
  printf '  "sha256": "%s",\n' "${SHA256}"
  printf '  "notes": "WakeWord Trainer macOS update %s."\n' "${VERSION_LABEL}"
  printf '}\n'
} > "${MANIFEST_PATH}"
cp "${MANIFEST_PATH}" "${REPO_MANIFEST_PATH}"

printf 'Packaged %s\n' "${ZIP_PATH}"
printf 'Copied %s\n' "${RELEASE_ZIP_PATH}"
printf 'Wrote %s\n' "${MANIFEST_PATH}"
printf 'Updated %s\n' "${REPO_MANIFEST_PATH}"
