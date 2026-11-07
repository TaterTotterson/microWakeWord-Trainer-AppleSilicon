#!/usr/bin/env bash
# Cross-machine issue claims for watcher.sh.
#
# Each claim is a temporary lightweight tag whose target commit contains the
# watcher identity and lease expiry. Git's ref creation and force-with-lease
# updates make acquire, renew, stale takeover, and release atomic across Macs.

set -u
set -o pipefail

ACTION="${1:-}"
CATALOG_REPO_DIR="${2:-}"
ISSUE_NUMBER="${3:-}"
WATCHER_ID="${4:-}"
TTL_SECONDS="${5:-0}"
CLAIM_PREFIX="${MW_CLAIM_PREFIX:-mww-watcher-claim}"
NOW_EPOCH="${MW_CLAIM_NOW_EPOCH:-$(date +%s)}"

usage() {
  printf 'Usage: %s <acquire|renew|release|status> <catalog-repo> <issue-number> <watcher-id> <ttl-seconds>\n' "$0" >&2
  exit 2
}

[[ "$ACTION" =~ ^(acquire|renew|release|status)$ ]] || usage
[[ -d "$CATALOG_REPO_DIR/.git" ]] || usage
[[ "$ISSUE_NUMBER" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$WATCHER_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]] || usage
[[ "$TTL_SECONDS" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$NOW_EPOCH" =~ ^[0-9]+$ ]] || usage
[[ "$CLAIM_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || usage

CLAIM_TAG="${CLAIM_PREFIX}-${ISSUE_NUMBER}"
CLAIM_REF="refs/tags/${CLAIM_TAG}"

read_claim() {
  local listing sha fetched_sha message parsed
  if ! listing="$(git -C "$CATALOG_REPO_DIR" ls-remote --refs origin "$CLAIM_REF" 2>/dev/null)"; then
    return 75
  fi
  sha="$(printf '%s\n' "$listing" | awk 'NR == 1 {print $1}')"
  [[ -n "$sha" ]] || return 4

  if ! git -C "$CATALOG_REPO_DIR" fetch --quiet origin "$CLAIM_REF" 2>/dev/null; then
    return 75
  fi
  fetched_sha="$(git -C "$CATALOG_REPO_DIR" rev-parse FETCH_HEAD 2>/dev/null || true)"
  [[ "$fetched_sha" == "$sha" ]] || return 75
  message="$(git -C "$CATALOG_REPO_DIR" show -s --format=%B "$sha" 2>/dev/null || true)"
  parsed="$(python3 - "$message" <<'PY'
import json
import sys

try:
    value = json.loads(sys.argv[1])
    owner = str(value["watcher_id"])
    expires_at = int(value["expires_at"])
except (KeyError, TypeError, ValueError, json.JSONDecodeError):
    owner = "unknown"
    expires_at = 0
print(f"{owner}\t{expires_at}")
PY
)"
  printf '%s\t%s\n' "$sha" "$parsed"
}

claim_payload() {
  local expires_at="$1"
  python3 - "$WATCHER_ID" "$ISSUE_NUMBER" "$NOW_EPOCH" "$expires_at" <<'PY'
import json
import sys

watcher_id, issue_number, claimed_at, expires_at = sys.argv[1:]
print(json.dumps({
    "type": "tater_wake_word_watcher_claim",
    "watcher_id": watcher_id,
    "issue_number": int(issue_number),
    "claimed_at": int(claimed_at),
    "expires_at": int(expires_at),
}, separators=(",", ":")))
PY
}

make_claim_commit() {
  local expires_at="$1" tree payload
  tree="$(git -C "$CATALOG_REPO_DIR" rev-parse 'HEAD^{tree}')" || return 75
  payload="$(claim_payload "$expires_at")" || return 75
  printf '%s\n' "$payload" | \
    GIT_AUTHOR_NAME="tater-wake-word-watcher" \
    GIT_AUTHOR_EMAIL="tater-wake-word-watcher@users.noreply.github.com" \
    GIT_COMMITTER_NAME="tater-wake-word-watcher" \
    GIT_COMMITTER_EMAIL="tater-wake-word-watcher@users.noreply.github.com" \
    git -C "$CATALOG_REPO_DIR" commit-tree "$tree" -p HEAD 2>/dev/null
}

print_claim() {
  local status="$1" owner="$2" expires_at="$3"
  printf '%s\t%s\t%s\t%s\n' "$status" "$ISSUE_NUMBER" "$owner" "$expires_at"
}

create_claim() {
  local expires_at claim_commit existing rc
  expires_at=$((NOW_EPOCH + TTL_SECONDS))
  claim_commit="$(make_claim_commit "$expires_at")" || return 75
  if git -C "$CATALOG_REPO_DIR" push --quiet origin \
      "$claim_commit:$CLAIM_REF" 2>/dev/null; then
    print_claim "acquired" "$WATCHER_ID" "$expires_at"
    return 0
  fi

  existing="$(read_claim)"
  rc=$?
  if [[ $rc -eq 0 ]]; then
    IFS=$'\t' read -r _ owner current_expiry <<< "$existing"
    print_claim "busy" "$owner" "$current_expiry"
    return 76
  fi
  return 75
}

acquire_claim() {
  local existing rc sha owner expires_at
  existing="$(read_claim)"
  rc=$?
  if [[ $rc -eq 4 ]]; then
    create_claim
    return $?
  fi
  [[ $rc -eq 0 ]] || return 75
  IFS=$'\t' read -r sha owner expires_at <<< "$existing"

  if [[ "$owner" == "$WATCHER_ID" && "$expires_at" -gt "$NOW_EPOCH" ]]; then
    # A restarted watcher reuses its stable identity. Renew immediately so a
    # nearly expired lease cannot lapse before the first heartbeat.
    renew_claim
    return $?
  fi
  if [[ "$expires_at" -gt "$NOW_EPOCH" ]]; then
    print_claim "busy" "$owner" "$expires_at"
    return 76
  fi

  # Remove an expired lease only if the ref still points to the SHA inspected
  # above. A simultaneous renew or takeover makes this delete fail safely.
  if ! git -C "$CATALOG_REPO_DIR" push --quiet \
      --force-with-lease="$CLAIM_REF:$sha" origin ":$CLAIM_REF" 2>/dev/null; then
    existing="$(read_claim)"
    rc=$?
    if [[ $rc -eq 0 ]]; then
      IFS=$'\t' read -r _ owner expires_at <<< "$existing"
      print_claim "busy" "$owner" "$expires_at"
      return 76
    fi
    [[ $rc -eq 4 ]] || return 75
  fi
  create_claim
}

renew_claim() {
  local existing rc sha owner expires_at new_expiry claim_commit
  existing="$(read_claim)"
  rc=$?
  if [[ $rc -eq 75 ]]; then
    return 75
  fi
  [[ $rc -eq 0 ]] || return 76
  IFS=$'\t' read -r sha owner expires_at <<< "$existing"
  if [[ "$owner" != "$WATCHER_ID" ]]; then
    print_claim "lost" "$owner" "$expires_at"
    return 76
  fi

  new_expiry=$((NOW_EPOCH + TTL_SECONDS))
  claim_commit="$(make_claim_commit "$new_expiry")" || return 75
  if git -C "$CATALOG_REPO_DIR" push --quiet \
      --force-with-lease="$CLAIM_REF:$sha" origin \
      "$claim_commit:$CLAIM_REF" 2>/dev/null; then
    print_claim "renewed" "$WATCHER_ID" "$new_expiry"
    return 0
  fi

  # A rejected compare-and-swap means the claim changed. A network failure is
  # different: report it as temporarily unavailable so watcher.sh can keep the
  # current job alive while the existing lease remains safely in force.
  existing="$(read_claim)"
  rc=$?
  [[ $rc -eq 75 ]] && return 75
  [[ $rc -eq 0 ]] || return 76
  IFS=$'\t' read -r _ owner expires_at <<< "$existing"
  if [[ "$owner" != "$WATCHER_ID" ]]; then
    print_claim "lost" "$owner" "$expires_at"
    return 76
  fi
  return 75
}

release_claim() {
  local existing rc sha owner expires_at
  existing="$(read_claim)"
  rc=$?
  if [[ $rc -eq 4 ]]; then
    print_claim "released" "$WATCHER_ID" 0
    return 0
  fi
  [[ $rc -eq 0 ]] || return 75
  IFS=$'\t' read -r sha owner expires_at <<< "$existing"
  if [[ "$owner" != "$WATCHER_ID" ]]; then
    print_claim "not-owner" "$owner" "$expires_at"
    return 0
  fi
  if git -C "$CATALOG_REPO_DIR" push --quiet \
      --force-with-lease="$CLAIM_REF:$sha" origin ":$CLAIM_REF" 2>/dev/null; then
    print_claim "released" "$WATCHER_ID" 0
    return 0
  fi
  return 75
}

status_claim() {
  local existing rc sha owner expires_at
  existing="$(read_claim)"
  rc=$?
  if [[ $rc -eq 4 ]]; then
    print_claim "none" "" 0
    return 0
  fi
  [[ $rc -eq 0 ]] || return 75
  IFS=$'\t' read -r sha owner expires_at <<< "$existing"
  print_claim "claimed" "$owner" "$expires_at"
}

case "$ACTION" in
  acquire) acquire_claim ;;
  renew) renew_claim ;;
  release) release_claim ;;
  status) status_claim ;;
esac
