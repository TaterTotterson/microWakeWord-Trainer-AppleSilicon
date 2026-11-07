#!/usr/bin/env bash
# Poll Tater-Wake-Words for wake-word requests and train them locally.
#
# This process is intentionally independent of GitHub Actions. A temporary
# network outage only pauses polling or publishing; it does not cancel a local
# training process. Requests carrying mww-processing are eligible so an issue
# interrupted by a watcher or Mac restart is reclaimed on the next poll.

set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="${PATH:-}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

MW_ISSUE_REPO="${MW_ISSUE_REPO:-TaterTotterson/Tater-Wake-Words}"
MW_CATALOG_REPO="${MW_CATALOG_REPO:-$MW_ISSUE_REPO}"
MW_CATALOG_BRANCH="${MW_CATALOG_BRANCH:-main}"
MW_CATALOG_DIR="${MW_CATALOG_DIR:-microWakeWordsV9}"
MW_POLL_SECONDS="${MW_POLL_SECONDS:-60}"
MW_LABEL_PROCESSING="${MW_LABEL_PROCESSING:-mww-processing}"
MW_LABEL_DONE="${MW_LABEL_DONE:-mww-added}"
MW_LABEL_FAILED="${MW_LABEL_FAILED:-mww-failed}"
MW_RETRY_FAILED="${MW_RETRY_FAILED:-0}"
MW_DRY_RUN="${MW_DRY_RUN:-0}"
MW_ONCE="${MW_ONCE:-0}"
MW_DEBUG="${MW_DEBUG:-0}"
MW_WATCHER_ID="${MW_WATCHER_ID:-}"
MW_CLAIM_TTL_SECONDS="${MW_CLAIM_TTL_SECONDS:-86400}"
MW_CLAIM_REFRESH_SECONDS="${MW_CLAIM_REFRESH_SECONDS:-300}"

MW_EXTERNAL_ROOT="${TATER_WAKE_EXTERNAL_ROOT:-$(dirname "$ROOT_DIR")}"
MW_TMP_ROOT="${TATER_WAKE_TMP_ROOT:-${MW_EXTERNAL_ROOT}/tater-wake-tmp}"
MW_STATE_DIR="${MW_WATCHER_STATE_DIR:-${MW_TMP_ROOT}/watcher}"
MW_CATALOG_REPO_DIR="${MW_CATALOG_REPO_DIR:-${MW_TMP_ROOT}/repos/Tater-Wake-Words}"
MW_TRAINER_DATA_DIR="${TATER_WAKE_DATA_DIR:-$ROOT_DIR}"
MW_TRAINER_SUPPORT_DIR="${TATER_WAKE_SUPPORT_DIR:-$ROOT_DIR}"

if [[ -n "${MW_GITHUB_TOKEN:-}" ]]; then
  export GH_TOKEN="$MW_GITHUB_TOKEN"
fi
if [[ -n "${GH_TOKEN:-}" ]]; then
  export GITHUB_TOKEN="${GITHUB_TOKEN:-$GH_TOKEN}"
fi

log() {
  printf "%s [mw_watcher] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

debug() {
  [[ "$MW_DEBUG" == "1" ]] && log "DEBUG: $*"
  return 0
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    log "ERROR: Required command is missing: $1"
    exit 1
  }
}

for command_name in gh git python3; do
  require_command "$command_name"
done

# Configure Git to obtain HTTPS credentials from the same gh login or GH_TOKEN
# used for issue polling. This keeps a token in .env out of remote URLs and logs.
if ! gh auth setup-git >/dev/null 2>&1; then
  log "WARNING: GitHub authentication is not ready; polling will retry until gh auth login or MW_GITHUB_TOKEN is configured"
fi

mkdir -p "$MW_STATE_DIR" "$MW_TMP_ROOT/tmp" "$(dirname "$MW_CATALOG_REPO_DIR")"

if [[ ! "$MW_CLAIM_TTL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    [[ "$MW_CLAIM_TTL_SECONDS" -lt 600 ]]; then
  log "ERROR: MW_CLAIM_TTL_SECONDS must be at least 600"
  exit 1
fi
if [[ ! "$MW_CLAIM_REFRESH_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    [[ "$MW_CLAIM_REFRESH_SECONDS" -ge "$MW_CLAIM_TTL_SECONDS" ]]; then
  log "ERROR: MW_CLAIM_REFRESH_SECONDS must be positive and shorter than the claim TTL"
  exit 1
fi

if [[ -z "$MW_WATCHER_ID" ]]; then
  watcher_token_file="$MW_STATE_DIR/watcher-id"
  watcher_token=""
  if [[ -f "$watcher_token_file" ]]; then
    watcher_token="$(tr -cd 'A-Za-z0-9' < "$watcher_token_file")"
  fi
  if [[ -z "$watcher_token" ]]; then
    watcher_token="$(python3 -c 'import secrets; print(secrets.token_hex(8))')"
    printf '%s\n' "$watcher_token" > "$watcher_token_file"
    chmod 600 "$watcher_token_file"
  fi
  watcher_host="$(hostname -s 2>/dev/null || printf 'mac')"
  watcher_host="$(printf '%s' "$watcher_host" | tr -cs 'A-Za-z0-9._-' '-')"
  MW_WATCHER_ID="${watcher_host}-${watcher_token}"
fi
if [[ ! "$MW_WATCHER_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$ ]]; then
  log "ERROR: MW_WATCHER_ID must use 1-96 letters, numbers, dots, underscores, or hyphens"
  exit 1
fi

LOCK_DIR="$MW_STATE_DIR/lock"
CLAIM_TOOL="$ROOT_DIR/watcher_claims.sh"
CHILD_PID=""
CLAIM_HEARTBEAT_PID=""
CURRENT_CLAIM_ISSUE=""
STOP_REQUESTED=0
WATCHER_PID=$$

if [[ ! -f "$CLAIM_TOOL" ]]; then
  log "ERROR: Cross-Mac claim helper is missing: $CLAIM_TOOL"
  exit 1
fi

release_lock() {
  if [[ -d "$LOCK_DIR" ]]; then
    rm -f "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" >/dev/null 2>&1 || true
  fi
}

request_stop() {
  STOP_REQUESTED=1
  log "Stop requested; waiting for the active training command to exit"
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" >/dev/null 2>&1; then
    kill -TERM "$CHILD_PID" >/dev/null 2>&1 || true
  fi
}

stop_claim_heartbeat() {
  if [[ -n "$CLAIM_HEARTBEAT_PID" ]] && \
      kill -0 "$CLAIM_HEARTBEAT_PID" >/dev/null 2>&1; then
    kill -TERM "$CLAIM_HEARTBEAT_PID" >/dev/null 2>&1 || true
    wait "$CLAIM_HEARTBEAT_PID" >/dev/null 2>&1 || true
  fi
  CLAIM_HEARTBEAT_PID=""
}

release_current_claim() {
  stop_claim_heartbeat
  [[ -n "$CURRENT_CLAIM_ISSUE" ]] || return 0
  local release_output=""
  if release_output="$(/bin/bash "$CLAIM_TOOL" release \
      "$MW_CATALOG_REPO_DIR" "$CURRENT_CLAIM_ISSUE" \
      "$MW_WATCHER_ID" "$MW_CLAIM_TTL_SECONDS" 2>/dev/null)"; then
    debug "Released claim for #$CURRENT_CLAIM_ISSUE: $release_output"
  else
    log "WARNING: Could not release claim for #$CURRENT_CLAIM_ISSUE; its lease will expire automatically"
  fi
  CURRENT_CLAIM_ISSUE=""
}

on_exit() {
  release_current_claim
  release_lock
}

trap request_stop INT TERM
trap 'log "Ignoring SIGHUP; watcher remains active"' HUP
trap on_exit EXIT

acquire_lock() {
  if mkdir "$LOCK_DIR" >/dev/null 2>&1; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    return 0
  fi

  local old_pid=""
  if [[ -f "$LOCK_DIR/pid" ]]; then
    old_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  fi
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" >/dev/null 2>&1; then
    log "ERROR: Another watcher is already running with PID $old_pid"
    exit 1
  fi

  log "Removing stale watcher lock"
  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" >/dev/null 2>&1 || {
    log "ERROR: Could not remove stale lock: $LOCK_DIR"
    exit 1
  }
  mkdir "$LOCK_DIR"
  printf '%s\n' "$$" > "$LOCK_DIR/pid"
}

acquire_lock

start_claim_heartbeat() {
  local issue_number="$1"
  local initial_expiry=$(( $(date +%s) + MW_CLAIM_TTL_SECONDS ))
  (
    local heartbeat_sleep_pid=""
    local lease_expires="$initial_expiry"
    local renew_output renew_rc now

    stop_heartbeat_loop() {
      if [[ -n "$heartbeat_sleep_pid" ]] && \
          kill -0 "$heartbeat_sleep_pid" >/dev/null 2>&1; then
        kill -TERM "$heartbeat_sleep_pid" >/dev/null 2>&1 || true
        wait "$heartbeat_sleep_pid" >/dev/null 2>&1 || true
      fi
      exit 0
    }

    # Do not inherit the watcher's EXIT/HUP behavior in this helper process.
    # Its TERM handler also reaps the active sleep so it cannot hold open the
    # watcher's output pipes after a job completes.
    trap - EXIT HUP
    trap stop_heartbeat_loop INT TERM

    while true; do
      sleep "$MW_CLAIM_REFRESH_SECONDS" &
      heartbeat_sleep_pid=$!
      wait "$heartbeat_sleep_pid" || exit 0
      heartbeat_sleep_pid=""
      renew_output=""
      renew_rc=0
      renew_output="$(/bin/bash "$CLAIM_TOOL" renew \
        "$MW_CATALOG_REPO_DIR" "$issue_number" \
        "$MW_WATCHER_ID" "$MW_CLAIM_TTL_SECONDS" 2>/dev/null)" || renew_rc=$?
      now="$(date +%s)"
      if [[ $renew_rc -eq 0 ]]; then
        lease_expires=$((now + MW_CLAIM_TTL_SECONDS))
        debug "Renewed claim for #$issue_number: $renew_output"
        continue
      fi
      if [[ $renew_rc -eq 76 ]]; then
        log "ERROR: Claim for #$issue_number was lost; stopping this watcher before another Mac trains the same issue"
        kill -TERM "$WATCHER_PID" >/dev/null 2>&1 || true
        exit 0
      fi
      log "WARNING: Could not renew claim for #$issue_number; will retry while the current lease is safe"
      if [[ $((now + MW_CLAIM_REFRESH_SECONDS + 60)) -ge $lease_expires ]]; then
        log "ERROR: Claim for #$issue_number is nearing expiry; stopping to prevent duplicate training"
        kill -TERM "$WATCHER_PID" >/dev/null 2>&1 || true
        exit 0
      fi
    done
  ) &
  CLAIM_HEARTBEAT_PID=$!
}

sleep_until_next_poll() {
  local remaining="$MW_POLL_SECONDS"
  while [[ $remaining -gt 0 && $STOP_REQUESTED -eq 0 ]]; do
    if [[ $remaining -gt 5 ]]; then
      sleep 5
      remaining=$((remaining - 5))
    else
      sleep "$remaining"
      remaining=0
    fi
  done
}

git_counts() {
  git -C "$MW_CATALOG_REPO_DIR" rev-list --left-right --count \
    "HEAD...origin/${MW_CATALOG_BRANCH}" 2>/dev/null || printf '0\t0\n'
}

catalog_git_path_exists() {
  local git_dir
  git_dir="$(git -C "$MW_CATALOG_REPO_DIR" rev-parse --absolute-git-dir)"
  [[ -e "$git_dir/$1" ]]
}

finish_catalog_merge() {
  catalog_git_path_exists MERGE_HEAD || return 1

  local conflicts conflict
  conflicts="$(git -C "$MW_CATALOG_REPO_DIR" diff --name-only --diff-filter=U)"
  if [[ -n "$conflicts" ]]; then
    while IFS= read -r conflict; do
      [[ -n "$conflict" ]] || continue
      if [[ "$conflict" != "wake_word_manifest.json" ]]; then
        log "Catalog merge has a non-generated conflict in $conflict"
        return 1
      fi
    done <<< "$conflicts"
    log "Resolving the generated wake-word manifest after catalog merge"
  fi

  (
    cd "$MW_CATALOG_REPO_DIR" || exit 1
    python3 scripts/generate_wake_word_manifest.py
    git add wake_word_manifest.json
    if [[ -n "$(git diff --name-only --diff-filter=U)" ]]; then
      exit 1
    fi
    GIT_EDITOR=true git commit --no-edit
  )
}

recover_interrupted_git_operation() {
  if catalog_git_path_exists rebase-merge || \
      catalog_git_path_exists rebase-apply; then
    log "Aborting an interrupted catalog rebase from an earlier watcher run"
    if ! git -C "$MW_CATALOG_REPO_DIR" rebase --abort; then
      log "Could not abort the interrupted catalog rebase; will retry"
      return 1
    fi
  fi

  if catalog_git_path_exists MERGE_HEAD; then
    log "Finishing an interrupted catalog merge"
    if ! finish_catalog_merge; then
      git -C "$MW_CATALOG_REPO_DIR" merge --abort >/dev/null 2>&1 || true
      log "Could not safely finish the interrupted catalog merge; will retry"
      return 1
    fi
  fi
  return 0
}

sync_catalog_checkout() {
  if [[ ! -d "$MW_CATALOG_REPO_DIR/.git" ]]; then
    log "Cloning $MW_CATALOG_REPO into $MW_CATALOG_REPO_DIR"
    if ! gh repo clone "$MW_CATALOG_REPO" "$MW_CATALOG_REPO_DIR" -- \
      --branch "$MW_CATALOG_BRANCH" --single-branch; then
      log "GitHub is unavailable or the catalog clone failed; will retry"
      return 1
    fi
  fi

  git -C "$MW_CATALOG_REPO_DIR" config user.name "tater-wake-word-watcher"
  git -C "$MW_CATALOG_REPO_DIR" config user.email \
    "tater-wake-word-watcher@users.noreply.github.com"

  if ! recover_interrupted_git_operation; then
    return 1
  fi

  if ! git -C "$MW_CATALOG_REPO_DIR" fetch --quiet origin "$MW_CATALOG_BRANCH"; then
    log "Could not refresh the catalog repository; will retry"
    return 1
  fi

  if [[ -n "$(git -C "$MW_CATALOG_REPO_DIR" status --porcelain)" ]]; then
    # This is a watcher-owned checkout. Dirty catalog files can be the complete or
    # partial output of a run whose connection failed before commit/push. The
    # matching issue will validate and recover them instead of deleting them.
    log "Catalog checkout contains interrupted local work; preserving it for recovery"
    return 0
  fi

  local counts ahead behind
  counts="$(git_counts)"
  ahead="$(printf '%s' "$counts" | awk '{print $1}')"
  behind="$(printf '%s' "$counts" | awk '{print $2}')"

  if [[ "$behind" -gt 0 && "$ahead" -eq 0 ]]; then
    log "Updating catalog checkout from origin/$MW_CATALOG_BRANCH"
    if ! git -C "$MW_CATALOG_REPO_DIR" merge --ff-only \
      "origin/$MW_CATALOG_BRANCH"; then
      log "Catalog update could not be fast-forwarded; will retry"
      return 1
    fi
  elif [[ "$behind" -gt 0 ]]; then
    # Each wake-word commit regenerates wake_word_manifest.json, so two valid
    # additions commonly conflict when a local commit could not be pushed and
    # main advanced independently. Merge the histories and regenerate that
    # derived file from the combined artifacts instead of rebasing it.
    log "Merging origin/$MW_CATALOG_BRANCH into recovered local catalog commits"
    if ! git -C "$MW_CATALOG_REPO_DIR" merge --no-ff --no-commit --no-edit \
      "origin/$MW_CATALOG_BRANCH"; then
      if ! finish_catalog_merge; then
        git -C "$MW_CATALOG_REPO_DIR" merge --abort >/dev/null 2>&1 || true
        log "Catalog update has conflicts outside the generated manifest; will retry"
        return 1
      fi
    elif ! finish_catalog_merge; then
      git -C "$MW_CATALOG_REPO_DIR" merge --abort >/dev/null 2>&1 || true
      log "Catalog merge could not regenerate its manifest; will retry"
      return 1
    fi
  fi

  counts="$(git_counts)"
  ahead="$(printf '%s' "$counts" | awk '{print $1}')"
  if [[ "$ahead" -gt 0 ]]; then
    log "Publishing $ahead recovered catalog commit(s)"
    if ! git -C "$MW_CATALOG_REPO_DIR" push origin \
      "HEAD:${MW_CATALOG_BRANCH}"; then
      log "Catalog push failed; keeping commits locally for the next poll"
      return 1
    fi
  fi
  return 0
}

list_candidates() {
  local issues_file="$MW_STATE_DIR/issues.json"
  if ! gh issue list \
    --repo "$MW_ISSUE_REPO" \
    --state open \
    --limit 1000 \
    --json number,title,labels,createdAt > "$issues_file"; then
    log "GitHub issue lookup failed; will retry after the connection returns"
    return 1
  fi

  MW_LABEL_DONE="$MW_LABEL_DONE" \
  MW_LABEL_FAILED="$MW_LABEL_FAILED" \
  MW_RETRY_FAILED="$MW_RETRY_FAILED" \
  MW_CATALOG_REPO_DIR="$MW_CATALOG_REPO_DIR" \
  MW_CATALOG_DIR="$MW_CATALOG_DIR" \
    python3 - "$issues_file" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path


def safe_slug(phrase: str) -> str:
    normalized = unicodedata.normalize("NFKC", phrase).strip().lower()
    slug = re.sub(r"[^a-z0-9_]+", "", re.sub(r"\s+", "_", normalized)).strip("_")
    if slug:
        return slug
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"wakeword_{digest}"


done_label = os.environ["MW_LABEL_DONE"]
failed_label = os.environ["MW_LABEL_FAILED"]
retry_failed = os.environ.get("MW_RETRY_FAILED") == "1"
catalog_repo_dir = Path(os.environ["MW_CATALOG_REPO_DIR"])
catalog_dir = os.environ["MW_CATALOG_DIR"]
issues = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

for issue in sorted(issues, key=lambda item: (item.get("createdAt", ""), item.get("number", 0))):
    match = re.match(r"^\s*mww:\s*(.+?)\s*$", str(issue.get("title") or ""), flags=re.I)
    if not match:
        continue
    labels = {
        str(label.get("name") or "")
        for label in issue.get("labels") or []
        if isinstance(label, dict)
    }
    if done_label in labels:
        continue

    phrase = match.group(1).strip().replace("\t", " ").replace("\n", " ")
    if phrase:
        slug = safe_slug(phrase)
        base = catalog_repo_dir / catalog_dir / slug
        package_complete = all(
            Path(f"{base}{suffix}").is_file()
            for suffix in (
                ".wake-bundle.json",
                ".esphome.json",
                ".tflite",
                ".oww.onnx",
                ".nww.cnn.onnx",
                ".nww.gru.onnx",
                ".boundary.tflite",
            )
        )
        # A publish failure can label the issue failed after all training has
        # completed. Reclaim that request when its complete local package is
        # present so the watcher can publish it and close the issue without
        # retraining. Other failed requests still require MW_RETRY_FAILED=1.
        if failed_label in labels and not retry_failed and not package_complete:
            continue
        print(f"{int(issue['number'])}\t{phrase}\t{slug}")
PY
}

catalog_package_is_complete() {
  local safe_word="$1"
  local base="$MW_CATALOG_REPO_DIR/$MW_CATALOG_DIR/$safe_word"
  local suffix
  for suffix in .wake-bundle.json .esphome.json .tflite .oww.onnx .nww.cnn.onnx .nww.gru.onnx .boundary.tflite; do
    [[ -f "${base}${suffix}" ]] || return 1
  done

  (
    cd "$MW_CATALOG_REPO_DIR" || exit 1
    python3 - "$MW_CATALOG_DIR/$safe_word.wake-bundle.json" <<'PY'
from pathlib import Path
import sys

from scripts.generate_wake_word_manifest import bundle_model_fields

raise SystemExit(
    0 if bundle_model_fields(Path(sys.argv[1])) else 1
)
PY
  )
}

recover_catalog_package() {
  local safe_word="$1"
  catalog_package_is_complete "$safe_word" || return 0

  local base="$MW_CATALOG_DIR/$safe_word"
  (
    cd "$MW_CATALOG_REPO_DIR" || exit 1
    python3 scripts/generate_wake_word_manifest.py
    git add \
      "${base}.wake-bundle.json" \
      "${base}.esphome.json" \
      "${base}.tflite" \
      "${base}.oww.onnx" \
      "${base}.nww.cnn.onnx" \
      "${base}.nww.gru.onnx" \
      "${base}.boundary.tflite" \
      wake_word_manifest.json
    if ! git diff --cached --quiet; then
      log "Recovering completed local artifacts for $safe_word"
      git commit -m "Add wake word: $safe_word"
    fi
    git push origin "HEAD:${MW_CATALOG_BRANCH}"
  )
}

run_claimed_issue() {
  local issue_number="$1" phrase="$2" safe_word="$3"
  local event_file="$MW_STATE_DIR/issue-${issue_number}.json"

  if ! recover_catalog_package "$safe_word"; then
    log "Could not publish recovered artifacts for #$issue_number; will retry"
    return 75
  fi

  printf '{"issue":{"number":%s}}\n' "$issue_number" > "$event_file"
  log "Processing #$issue_number: '$phrase' (artifact: $safe_word)"

  (
    cd "$MW_CATALOG_REPO_DIR" || exit 1
    export GITHUB_REPOSITORY="$MW_ISSUE_REPO"
    export GITHUB_EVENT_PATH="$event_file"
    export WAKE_WORD_ISSUE_NUMBER="$issue_number"
    export CATALOG_DIR="$MW_CATALOG_DIR"
    export LABEL_PROCESSING="$MW_LABEL_PROCESSING"
    export LABEL_DONE="$MW_LABEL_DONE"
    export LABEL_FAILED="$MW_LABEL_FAILED"
    export TATER_WAKE_EXTERNAL_ROOT="$MW_EXTERNAL_ROOT"
    export TATER_WAKE_TRAINER_DIR="$ROOT_DIR"
    export TATER_WAKE_TMP_ROOT="$MW_TMP_ROOT"
    export TATER_WAKE_DATA_DIR="$MW_TRAINER_DATA_DIR"
    export TATER_WAKE_SUPPORT_DIR="$MW_TRAINER_SUPPORT_DIR"
    export RUNNER_TEMP="$MW_TMP_ROOT/tmp"
    exec /bin/bash scripts/train_issue_wake_word.sh
  ) &
  CHILD_PID=$!

  local train_rc=0
  wait "$CHILD_PID" || train_rc=$?
  CHILD_PID=""
  rm -f "$event_file"

  if [[ $train_rc -eq 0 ]]; then
    log "Completed #$issue_number ($safe_word)"
  else
    log "Training #$issue_number exited with status $train_rc; the issue remains open"
  fi
  return "$train_rc"
}

run_issue() {
  local issue_number="$1" phrase="$2" safe_word="$3"
  local claim_output="" claim_rc=0 issue_rc=0

  if ! sync_catalog_checkout; then
    return 75
  fi

  claim_output="$(/bin/bash "$CLAIM_TOOL" acquire \
    "$MW_CATALOG_REPO_DIR" "$issue_number" \
    "$MW_WATCHER_ID" "$MW_CLAIM_TTL_SECONDS" 2>/dev/null)" || claim_rc=$?
  if [[ $claim_rc -eq 76 ]]; then
    log "Skipping #$issue_number; another Mac already holds its claim ($claim_output)"
    if [[ -n "$(git -C "$MW_CATALOG_REPO_DIR" status --porcelain)" ]]; then
      log "Interrupted local catalog work is still present; deferring other requests until it can be recovered safely"
      return 75
    fi
    return 76
  fi
  if [[ $claim_rc -ne 0 ]]; then
    log "Could not claim #$issue_number; GitHub or the catalog remote is unavailable"
    return 75
  fi

  CURRENT_CLAIM_ISSUE="$issue_number"
  log "Claimed #$issue_number as $MW_WATCHER_ID"
  start_claim_heartbeat "$issue_number"
  run_claimed_issue "$issue_number" "$phrase" "$safe_word" || issue_rc=$?
  release_current_claim
  return "$issue_rc"
}

log "Starting independent wake-word issue watcher"
log "Issue repo:       $MW_ISSUE_REPO"
log "Catalog checkout: $MW_CATALOG_REPO_DIR"
log "Catalog path:     $MW_CATALOG_DIR"
log "Trainer:          $ROOT_DIR"
log "Training data:    $MW_TRAINER_DATA_DIR"
log "Watcher ID:       $MW_WATCHER_ID"
log "Claim lease:      ${MW_CLAIM_TTL_SECONDS}s (refresh ${MW_CLAIM_REFRESH_SECONDS}s)"
log "Poll interval:    ${MW_POLL_SECONDS}s"
log "Retry failed:     $MW_RETRY_FAILED"
log "Dry run:          $MW_DRY_RUN"

while [[ $STOP_REQUESTED -eq 0 ]]; do
  candidates_file="$MW_STATE_DIR/candidates.tsv"
  if ! list_candidates > "$candidates_file"; then
    [[ "$MW_ONCE" == "1" ]] && break
    sleep_until_next_poll
    continue
  fi

  if [[ ! -s "$candidates_file" ]]; then
    debug "No pending wake-word requests"
  else
    while IFS=$'\t' read -r issue_number phrase safe_word; do
      [[ -n "${issue_number:-}" && -n "${phrase:-}" && -n "${safe_word:-}" ]] || continue
      [[ $STOP_REQUESTED -eq 0 ]] || break

      if [[ "$MW_DRY_RUN" == "1" ]]; then
        log "DRY RUN: would process #$issue_number '$phrase' as '$safe_word'"
        continue
      fi

      run_rc=0
      run_issue "$issue_number" "$phrase" "$safe_word" || run_rc=$?
      if [[ $run_rc -eq 75 ]]; then
        log "Catalog or GitHub is temporarily unavailable; deferring the remaining requests"
        break
      fi
    done < "$candidates_file"
  fi

  [[ "$MW_ONCE" == "1" ]] && break
  sleep_until_next_poll
done

log "Watcher stopped"
