#!/bin/sh
set -eu

ARTIFACT="${1:?Usage: notarize_artifact.sh /path/to/artifact}"

if [ "${WAKEWORD_TRAINER_NOTARIZE:-0}" != "1" ]; then
  printf 'Skipping notarization for %s (set WAKEWORD_TRAINER_NOTARIZE=1 to enable).\n' "${ARTIFACT}"
  exit 0
fi

if [ ! -e "${ARTIFACT}" ]; then
  printf 'Cannot notarize missing artifact: %s\n' "${ARTIFACT}" >&2
  exit 1
fi

if [ -z "${WAKEWORD_TRAINER_NOTARY_PROFILE:-}" ]; then
  printf 'WAKEWORD_TRAINER_NOTARIZE=1, but WAKEWORD_TRAINER_NOTARY_PROFILE was not configured.\n' >&2
  exit 1
fi

xcrun notarytool submit "${ARTIFACT}" --wait --keychain-profile "${WAKEWORD_TRAINER_NOTARY_PROFILE}"
