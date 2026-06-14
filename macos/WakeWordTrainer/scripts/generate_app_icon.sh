#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd -P)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
RESOURCES_DIR="${PROJECT_DIR}/Resources"
SOURCE_IMAGE="${RESOURCES_DIR}/WakeWordIconArtwork.png"
SOURCE_PREVIEW="${RESOURCES_DIR}/WakeWordIconSource.png"
ICONSET_DIR="${PROJECT_DIR}/build/WakeWordIcon.iconset"
OUTPUT_ICON="${RESOURCES_DIR}/WakeWordIcon.icns"

if [ ! -f "${SOURCE_IMAGE}" ]; then
  printf 'Missing icon source: %s\n' "${SOURCE_IMAGE}" >&2
  exit 1
fi

rm -rf "${ICONSET_DIR}"
mkdir -p "${ICONSET_DIR}"

sips -s format png -z 1024 1024 "${SOURCE_IMAGE}" --out "${SOURCE_PREVIEW}" >/dev/null

sips -s format png -z 16 16 "${SOURCE_IMAGE}" --out "${ICONSET_DIR}/icon_16x16.png" >/dev/null
sips -s format png -z 32 32 "${SOURCE_IMAGE}" --out "${ICONSET_DIR}/icon_16x16@2x.png" >/dev/null
sips -s format png -z 32 32 "${SOURCE_IMAGE}" --out "${ICONSET_DIR}/icon_32x32.png" >/dev/null
sips -s format png -z 64 64 "${SOURCE_IMAGE}" --out "${ICONSET_DIR}/icon_32x32@2x.png" >/dev/null
sips -s format png -z 128 128 "${SOURCE_IMAGE}" --out "${ICONSET_DIR}/icon_128x128.png" >/dev/null
sips -s format png -z 256 256 "${SOURCE_IMAGE}" --out "${ICONSET_DIR}/icon_128x128@2x.png" >/dev/null
sips -s format png -z 256 256 "${SOURCE_IMAGE}" --out "${ICONSET_DIR}/icon_256x256.png" >/dev/null
sips -s format png -z 512 512 "${SOURCE_IMAGE}" --out "${ICONSET_DIR}/icon_256x256@2x.png" >/dev/null
sips -s format png -z 512 512 "${SOURCE_IMAGE}" --out "${ICONSET_DIR}/icon_512x512.png" >/dev/null
sips -s format png -z 1024 1024 "${SOURCE_IMAGE}" --out "${ICONSET_DIR}/icon_512x512@2x.png" >/dev/null

iconutil -c icns "${ICONSET_DIR}" -o "${OUTPUT_ICON}"
