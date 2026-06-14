#!/bin/sh
set -eu
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(CDPATH= cd "$(dirname "$0")" && pwd -P)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
INFO_PLIST="${PROJECT_DIR}/Resources/Info.plist"
BACKGROUND_IMAGE="${PROJECT_DIR}/Resources/WakeWordDmgBackground.png"
APP_DIR="${PROJECT_DIR}/build/WakeWord Trainer.app"
RELEASES_DIR="${PROJECT_DIR}/releases"

VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "${INFO_PLIST}")"
VERSION_TOKEN="$(printf '%s' "${VERSION}" | sed 's/^[vV]//')"
VERSION_LABEL="v${VERSION_TOKEN}"
VOLUME_NAME="Install WakeWord Trainer ${VERSION_LABEL}"
DMG_NAME="WakeWordTrainer-${VERSION_LABEL}.dmg"
FINAL_DMG="${PROJECT_DIR}/build/${DMG_NAME}"
RELEASE_DMG="${RELEASES_DIR}/${DMG_NAME}"
RW_DMG="${PROJECT_DIR}/build/.${DMG_NAME%.dmg}.rw.dmg"
STAGING_DIR="${PROJECT_DIR}/build/dmg-staging"
MOUNT_DIR=""
DEVICE=""

cleanup() {
  if [ -n "${DEVICE}" ]; then
    hdiutil detach "${DEVICE}" -quiet >/dev/null 2>&1 || true
  fi
  rm -rf "${STAGING_DIR}" "${RW_DMG}"
}
trap cleanup EXIT INT TERM

if [ ! -f "${BACKGROUND_IMAGE}" ]; then
  printf 'Missing DMG background image: %s\n' "${BACKGROUND_IMAGE}" >&2
  exit 1
fi

"${SCRIPT_DIR}/build_app.sh"

rm -rf "${STAGING_DIR}" "${FINAL_DMG}" "${RW_DMG}"
mkdir -p "${STAGING_DIR}/.background"

ditto "${APP_DIR}" "${STAGING_DIR}/WakeWord Trainer.app"
ln -s /Applications "${STAGING_DIR}/Applications"
cp "${BACKGROUND_IMAGE}" "${STAGING_DIR}/.background/WakeWordDmgBackground.png"

hdiutil create \
  -volname "${VOLUME_NAME}" \
  -srcfolder "${STAGING_DIR}" \
  -fs HFS+ \
  -fsargs "-c c=64,a=16,e=16" \
  -format UDRW \
  -ov \
  "${RW_DMG}" >/dev/null

ATTACH_OUTPUT="$(hdiutil attach "${RW_DMG}" -readwrite -noverify -noautoopen)"
DEVICE="$(printf '%s\n' "${ATTACH_OUTPUT}" | awk '/Apple_HFS/ {print $1; exit}')"
MOUNT_DIR="$(printf '%s\n' "${ATTACH_OUTPUT}" | awk '/Apple_HFS/ {sub(/^.*Apple_HFS[[:space:]]+/, ""); print; exit}')"
if [ -z "${DEVICE}" ]; then
  printf 'Could not determine mounted DMG device.\n' >&2
  printf '%s\n' "${ATTACH_OUTPUT}" >&2
  exit 1
fi
if [ -z "${MOUNT_DIR}" ]; then
  printf 'Could not determine mounted DMG path.\n' >&2
  printf '%s\n' "${ATTACH_OUTPUT}" >&2
  exit 1
fi

VOLUME_NAME="$(basename "${MOUNT_DIR}")"

sleep 1

osascript <<APPLESCRIPT
tell application "Finder"
  tell disk "${VOLUME_NAME}"
    open
    set current view of container window to icon view
    set toolbar visible of container window to false
    set statusbar visible of container window to false
    try
      set pathbar visible of container window to false
    end try
    set bounds of container window to {120, 120, 888, 664}
    set viewOptions to the icon view options of container window
    set arrangement of viewOptions to not arranged
    set icon size of viewOptions to 96
    set text size of viewOptions to 12
    set background picture of viewOptions to file ".background:WakeWordDmgBackground.png"
    set position of item "WakeWord Trainer.app" of container window to {410, 250}
    set position of item "Applications" of container window to {615, 250}
    try
      set position of item ".background" of container window to {2000, 2000}
    end try
    try
      set position of item ".fseventsd" of container window to {2000, 2000}
    end try
    update without registering applications
    delay 1
    close
  end tell
end tell
APPLESCRIPT

sync
SetFile -a V "${MOUNT_DIR}/.background" >/dev/null 2>&1 || true
chflags hidden "${MOUNT_DIR}/.background" >/dev/null 2>&1 || true
if [ -d "${MOUNT_DIR}/.fseventsd" ]; then
  SetFile -a V "${MOUNT_DIR}/.fseventsd" >/dev/null 2>&1 || true
  chflags hidden "${MOUNT_DIR}/.fseventsd" >/dev/null 2>&1 || true
fi
sync
hdiutil detach "${DEVICE}" -quiet
DEVICE=""

hdiutil convert "${RW_DMG}" \
  -format UDZO \
  -imagekey zlib-level=9 \
  -o "${FINAL_DMG}" >/dev/null

hdiutil verify "${FINAL_DMG}" >/dev/null

mkdir -p "${RELEASES_DIR}"
cp "${FINAL_DMG}" "${RELEASE_DMG}"

printf 'Built %s\n' "${FINAL_DMG}"
printf 'Copied %s\n' "${RELEASE_DMG}"
