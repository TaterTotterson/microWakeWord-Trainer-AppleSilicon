#!/usr/bin/env bash
# watcher.sh  (mw_issue_watcher_macos.sh)
#
# Minimal GitHub issue watcher for microWakeWord-Trainer-AppleSilicon (macOS).
#
# What it does:
# - Polls GitHub Search API for open issues with titles like:  mww: hey dude
# - Labels them as "mww-processing"
# - Runs: ./train_microwakeword_macos.sh "hey_dude"
# - Uploads hey_dude.tflite + hey_dude.json to your model repo/path
# - Comments "added!", labels "mww-added", and closes the issue
#
# Requirements:
# - MW_GITHUB_TOKEN in env (PAT; needs issues + contents on the target repo)
#
# Optional env:
# - MW_ISSUE_REPO (default: TaterTotterson/microWakeWords)
# - MW_MODEL_REPO (default: MW_ISSUE_REPO)
# - MW_MODEL_BRANCH (default: main)
# - MW_MODEL_BASE_PATH (default: microWakeWords)
# - MW_POLL_SECONDS (default: 60)
# - MW_LABEL_PROCESSING (default: mww-processing)
# - MW_LABEL_DONE (default: mww-added)
# - MW_DONE_COMMENT (default: added!)
# - MW_DRY_RUN=1   (do not train/upload/close; just log)
# - MW_DEBUG=1     (extra logging)

set -euo pipefail

# -------------------- Bash sanity --------------------
# macOS /bin/bash is 3.2; we require bash 4+ (for mapfile, etc).
if [[ -z "${BASH_VERSINFO:-}" ]]; then
  echo "❌ This script must be run with bash, not sh/zsh."
  echo "   Try: /opt/homebrew/bin/bash ./watcher.sh"
  exit 1
fi
if (( BASH_VERSINFO[0] < 4 )); then
  echo "❌ Bash ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]} detected (too old)."
  echo "   Install modern bash and run with it:"
  echo "     brew install bash"
  echo "     /opt/homebrew/bin/bash ./watcher.sh"
  exit 1
fi

# -------------------- Optional .env auto-load --------------------
# Note: this is safe for simple KEY=VALUE lines. If you have spaces, quote them.
if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------- Config --------------------
: "${MW_GITHUB_TOKEN:?MW_GITHUB_TOKEN is required}"

MW_ISSUE_REPO="${MW_ISSUE_REPO:-TaterTotterson/microWakeWords}"
MW_MODEL_REPO="${MW_MODEL_REPO:-$MW_ISSUE_REPO}"
MW_MODEL_BRANCH="${MW_MODEL_BRANCH:-main}"
MW_MODEL_BASE_PATH="${MW_MODEL_BASE_PATH:-microWakeWordsV2}"

MW_POLL_SECONDS="${MW_POLL_SECONDS:-60}"

MW_LABEL_PROCESSING="${MW_LABEL_PROCESSING:-mww-processing}"
MW_LABEL_DONE="${MW_LABEL_DONE:-mww-added}"
MW_DONE_COMMENT="${MW_DONE_COMMENT:-added!}"

MW_DRY_RUN="${MW_DRY_RUN:-0}"
MW_DEBUG="${MW_DEBUG:-0}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/train_microwakeword_macos.sh}"
if [[ ! -x "$TRAIN_SCRIPT" ]]; then
  echo "❌ Training script not found or not executable: $TRAIN_SCRIPT"
  exit 1
fi

# Use python3 from PATH (does NOT need to be the venv used by training script)
PY_BIN="${PY_BIN:-python3}"
command -v "$PY_BIN" >/dev/null 2>&1 || { echo "❌ python not found: $PY_BIN"; exit 1; }

# -------------------- Helpers --------------------
log() { printf "%s [mw_watcher] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

safe_name() {
  local s="${1:-}"
  s="$(echo "$s" | tr '[:upper:]' '[:lower:]')"
  s="$(echo "$s" | sed -E 's/[[:space:]]+/_/g')"
  s="$(echo "$s" | sed -E 's/[^a-z0-9_]+//g')"
  s="$(echo "$s" | sed -E 's/^_+|_+$//g')"
  [[ -n "$s" ]] || s="wakeword"
  echo "$s"
}

# Curl wrapper:
# - writes body to a file
# - writes headers to a file
# - prints http_code to stdout
curl_to_files() {
  local method="$1"; shift
  local url="$1"; shift
  local body_file="$1"; shift
  local headers_file="$1"; shift
  local data="${1:-}"

  # Always send auth + accept
  if [[ -n "$data" ]]; then
    curl -sS -X "$method" \
      -H "Authorization: Bearer $MW_GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "Content-Type: application/json" \
      -H "User-Agent: mw-issue-watcher" \
      -D "$headers_file" \
      -o "$body_file" \
      -w "%{http_code}" \
      --data "$data" \
      "$url"
  else
    curl -sS -X "$method" \
      -H "Authorization: Bearer $MW_GITHUB_TOKEN" \
      -H "Accept: application/vnd.github+json" \
      -H "User-Agent: mw-issue-watcher" \
      -D "$headers_file" \
      -o "$body_file" \
      -w "%{http_code}" \
      "$url"
  fi
}

# Pull rate limit headers for debug
debug_rate_limit() {
  local headers_file="$1"
  local limit remaining reset
  limit="$(grep -i '^x-ratelimit-limit:' "$headers_file" | tail -n1 | awk '{print $2}' | tr -d '\r')"
  remaining="$(grep -i '^x-ratelimit-remaining:' "$headers_file" | tail -n1 | awk '{print $2}' | tr -d '\r')"
  reset="$(grep -i '^x-ratelimit-reset:' "$headers_file" | tail -n1 | awk '{print $2}' | tr -d '\r')"
  if [[ -n "${limit:-}" || -n "${remaining:-}" ]]; then
    log "DEBUG: rate_limit limit=${limit:-?} remaining=${remaining:-?} reset=${reset:-?}"
  fi
}

# Extract candidates from GitHub Search API response body file.
# Output: lines "issue_number<TAB>raw_phrase"
extract_candidates_tsv() {
  local body_file="$1"
  "$PY_BIN" - <<'PY' "$body_file"
import json, re, sys, pathlib

p = pathlib.Path(sys.argv[1])
raw = p.read_text(encoding="utf-8", errors="replace").strip()
if not raw:
    sys.exit(0)

data = json.loads(raw)
items = data.get("items") if isinstance(data, dict) else None
if not isinstance(items, list):
    sys.exit(0)

for it in items:
    if not isinstance(it, dict):
        continue
    # Search API items are always issues/PRs; restrict to actual issues.
    # If it has "pull_request" key, it's a PR.
    if it.get("pull_request"):
        continue

    num = it.get("number")
    title = (it.get("title") or "")
    labels = [l.get("name") for l in (it.get("labels") or []) if isinstance(l, dict)]

    m = re.match(r"^\s*mww:\s*(.+?)\s*$", title, flags=re.I)
    if not m:
        continue
    phrase = (m.group(1) or "").strip()
    if not phrase:
        continue

    # Skip if already processing or done
    if "mww-processing" in labels or "mww-added" in labels:
        continue

    # Emit
    if num is not None:
        sys.stdout.write(f"{num}\t{phrase}\n")
PY
}

# GitHub API actions (issues + contents)
add_labels() {
  local repo="$1" number="$2"; shift 2
  local labels_json
  labels_json="$("$PY_BIN" - <<'PY' "$@"
import json, sys
labels = sys.argv[1:]
labels = [l for l in labels if l]
print(json.dumps({"labels": labels}))
PY
)"
  local body_file headers_file http_code
  body_file="$(mktemp)"; headers_file="$(mktemp)"
  http_code="$(curl_to_files POST "https://api.github.com/repos/${repo}/issues/${number}/labels" "$body_file" "$headers_file" "$labels_json")"
  if [[ "$MW_DEBUG" == "1" ]]; then
    log "DEBUG: add_labels http=$http_code repo=$repo issue=$number labels='$*'"
    debug_rate_limit "$headers_file"
  fi
  rm -f "$body_file" "$headers_file"
  [[ "$http_code" == "200" || "$http_code" == "201" ]] || return 1
}

comment_issue() {
  local repo="$1" number="$2" body="$3"
  local payload
  payload="$("$PY_BIN" - <<PY
import json
print(json.dumps({"body": ${body@Q}}))
PY
)"
  local body_file headers_file http_code
  body_file="$(mktemp)"; headers_file="$(mktemp)"
  http_code="$(curl_to_files POST "https://api.github.com/repos/${repo}/issues/${number}/comments" "$body_file" "$headers_file" "$payload")"
  if [[ "$MW_DEBUG" == "1" ]]; then
    log "DEBUG: comment http=$http_code repo=$repo issue=$number"
    debug_rate_limit "$headers_file"
  fi
  rm -f "$body_file" "$headers_file"
  [[ "$http_code" == "200" || "$http_code" == "201" ]]
}

close_issue() {
  local repo="$1" number="$2"
  local body_file headers_file http_code
  body_file="$(mktemp)"; headers_file="$(mktemp)"
  http_code="$(curl_to_files PATCH "https://api.github.com/repos/${repo}/issues/${number}" "$body_file" "$headers_file" '{"state":"closed"}')"
  if [[ "$MW_DEBUG" == "1" ]]; then
    log "DEBUG: close_issue http=$http_code repo=$repo issue=$number"
    debug_rate_limit "$headers_file"
  fi
  rm -f "$body_file" "$headers_file"
  [[ "$http_code" == "200" ]]
}

# Content SHA lookup (for update vs create)
get_content_sha() {
  local repo="$1" path="$2" branch="$3"
  local body_file headers_file http_code
  body_file="$(mktemp)"; headers_file="$(mktemp)"

  http_code="$(curl_to_files GET "https://api.github.com/repos/${repo}/contents/${path}?ref=${branch}" "$body_file" "$headers_file")"

  if [[ "$MW_DEBUG" == "1" ]]; then
    log "DEBUG: get_content_sha http=$http_code path=$path"
  fi

  # 200 => exists; 404 => not found
  if [[ "$http_code" != "200" ]]; then
    rm -f "$body_file" "$headers_file"
    echo ""
    return 0
  fi

  "$PY_BIN" - <<'PY' "$body_file"
import json, sys, pathlib
raw = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
j = json.loads(raw)
print(j.get("sha","") or "")
PY

  rm -f "$body_file" "$headers_file"
}

put_file() {
  local repo="$1" branch="$2" path="$3" local_file="$4" message="$5"

  if [[ ! -f "$local_file" ]]; then
    log "❌ Missing local file to upload: $local_file"
    return 1
  fi

  local sha
  sha="$(get_content_sha "$repo" "$path" "$branch")"

  local content_b64
  content_b64="$("$PY_BIN" - <<PY
import base64
data=open(${local_file@Q},'rb').read()
print(base64.b64encode(data).decode('utf-8'))
PY
)"

  local payload
  payload="$("$PY_BIN" - <<PY
import json
payload={
  "message": ${message@Q},
  "content": ${content_b64@Q},
  "branch": ${branch@Q},
}
sha=${sha@Q}
if sha:
  payload["sha"]=sha
print(json.dumps(payload))
PY
)"

  local body_file headers_file http_code
  body_file="$(mktemp)"; headers_file="$(mktemp)"
  http_code="$(curl_to_files PUT "https://api.github.com/repos/${repo}/contents/${path}" "$body_file" "$headers_file" "$payload")"

  if [[ "$MW_DEBUG" == "1" ]]; then
    log "DEBUG: put_file http=$http_code repo=$repo path=$path file=$(basename "$local_file")"
    debug_rate_limit "$headers_file"
    if [[ "$http_code" != "200" && "$http_code" != "201" ]]; then
      log "DEBUG: put_file body_head='$(head -c 200 "$body_file" | tr -d '\n' | tr -d '\r')'"
    fi
  fi

  rm -f "$body_file" "$headers_file"
  [[ "$http_code" == "200" || "$http_code" == "201" ]]
}

# -------------------- Startup logs --------------------
log "Starting issue watcher"
log "Issue repo:  $MW_ISSUE_REPO"
log "Model repo:  $MW_MODEL_REPO"
log "Branch:      $MW_MODEL_BRANCH"
log "Base path:   $MW_MODEL_BASE_PATH"
log "Poll every:  ${MW_POLL_SECONDS}s"
log "Dry run:     $MW_DRY_RUN"
log "Debug:       $MW_DEBUG"
log "Train script:$TRAIN_SCRIPT"

# -------------------- Main loop --------------------
while true; do
  if [[ "$MW_DEBUG" == "1" ]]; then
    log "Polling GitHub (Search API) for open mww: issues…"
  fi

  body_file="$(mktemp)"
  headers_file="$(mktemp)"

  # Search API: find open issues with "mww:" in title.
  # NOTE: Search API is rate-limited to 30 req/min for authenticated requests.
  # Query: repo:OWNER/REPO is:issue is:open in:title "mww:"
  search_url="https://api.github.com/search/issues?q=repo:${MW_ISSUE_REPO}+is:issue+is:open+in:title+%22mww:%22&sort=created&order=asc&per_page=100"

  set +e
  http_code="$(curl_to_files GET "$search_url" "$body_file" "$headers_file")"
  curl_rc=$?
  set -e

  if [[ "$MW_DEBUG" == "1" ]]; then
    first_http="$(head -n 1 "$headers_file" | tr -d '\r')"
    bytes="$(wc -c <"$body_file" | tr -d ' ')"
    log "DEBUG: curl_rc=$curl_rc http=$http_code first_http='${first_http}' url='${search_url}' body_bytes=$bytes"
    debug_rate_limit "$headers_file"
    log "DEBUG: body_head='$(head -c 180 "$body_file" | tr -d '\n' | tr -d '\r')'"
  fi

  if [[ $curl_rc -ne 0 || "$http_code" != "200" ]]; then
    log "⚠️ GitHub search API failed (curl_rc=$curl_rc http=$http_code). Sleeping ${MW_POLL_SECONDS}s…"
    rm -f "$body_file" "$headers_file"
    sleep "$MW_POLL_SECONDS"
    continue
  fi

  # -------- Candidate extraction (THIS is where set +e / set -e goes) --------
  set +e
  candidates_text="$(extract_candidates_tsv "$body_file")"
  py_rc=$?
  set -e

  if [[ $py_rc -ne 0 ]]; then
    log "❌ Candidate extraction failed (python rc=$py_rc). Sleeping ${MW_POLL_SECONDS}s…"
    rm -f "$body_file" "$headers_file"
    sleep "$MW_POLL_SECONDS"
    continue
  fi

  rm -f "$body_file" "$headers_file"

  if [[ -z "${candidates_text:-}" ]]; then
    if [[ "$MW_DEBUG" == "1" ]]; then
      log "DEBUG: No matching issues found this poll."
    fi
    sleep "$MW_POLL_SECONDS"
    continue
  fi

  # Process oldest first
  while IFS=$'\t' read -r issue_number raw_phrase; do
    [[ -n "${issue_number:-}" && -n "${raw_phrase:-}" ]] || continue

    safe_word="$(safe_name "$raw_phrase")"
    log "Found request: #${issue_number} phrase='${raw_phrase}' safe='${safe_word}'"

    if [[ "$MW_DRY_RUN" == "1" ]]; then
      log "DRY_RUN: would label issue, train, publish, comment, close."
      continue
    fi

    # Label as processing first to avoid double pickup
    log "Labeling #${issue_number} -> ${MW_LABEL_PROCESSING}"
    if ! add_labels "$MW_ISSUE_REPO" "$issue_number" "$MW_LABEL_PROCESSING"; then
      log "⚠️ Failed to add processing label to #${issue_number}, skipping"
      continue
    fi

    # Train: ONLY safe word
    log "Training: $TRAIN_SCRIPT \"${safe_word}\""
    set +e
    ( cd "$ROOT_DIR" && "$TRAIN_SCRIPT" "$safe_word" )
    train_rc=$?
    set -e

    if [[ $train_rc -ne 0 ]]; then
      log "❌ Training failed for '${safe_word}' (rc=$train_rc). Commenting and leaving issue open."
      comment_issue "$MW_ISSUE_REPO" "$issue_number" "Training failed for '${safe_word}' (exit code ${train_rc}). Check runner logs." || true
      continue
    fi

    # Expect artifacts in repo root
    tflite_file="$ROOT_DIR/${safe_word}.tflite"
    json_file="$ROOT_DIR/${safe_word}.json"

    if [[ ! -f "$tflite_file" || ! -f "$json_file" ]]; then
      log "❌ Artifacts not found after training: ${tflite_file} / ${json_file}"
      comment_issue "$MW_ISSUE_REPO" "$issue_number" "Training finished but artifacts were not found for '${safe_word}'." || true
      continue
    fi

    # Publish to model repo/path
    if [[ -n "$MW_MODEL_BASE_PATH" ]]; then
      tflite_path="${MW_MODEL_BASE_PATH}/${safe_word}.tflite"
      json_path="${MW_MODEL_BASE_PATH}/${safe_word}.json"
    else
      tflite_path="${safe_word}.tflite"
      json_path="${safe_word}.json"
    fi

    msg="Add microWakeWord: ${safe_word}"

    log "Publishing to GitHub: ${MW_MODEL_REPO}@${MW_MODEL_BRANCH} -> ${tflite_path}, ${json_path}"
    if ! put_file "$MW_MODEL_REPO" "$MW_MODEL_BRANCH" "$tflite_path" "$tflite_file" "$msg"; then
      log "❌ Failed to upload tflite"
      comment_issue "$MW_ISSUE_REPO" "$issue_number" "Training succeeded but upload failed for '${safe_word}.tflite'." || true
      continue
    fi
    if ! put_file "$MW_MODEL_REPO" "$MW_MODEL_BRANCH" "$json_path" "$json_file" "$msg"; then
      log "❌ Failed to upload json"
      comment_issue "$MW_ISSUE_REPO" "$issue_number" "Training succeeded but upload failed for '${safe_word}.json'." || true
      continue
    fi

    # Comment + label done + close
    log "Commenting + labeling done + closing #${issue_number}"
    comment_issue "$MW_ISSUE_REPO" "$issue_number" "$MW_DONE_COMMENT" || true
    add_labels "$MW_ISSUE_REPO" "$issue_number" "$MW_LABEL_DONE" || true
    close_issue "$MW_ISSUE_REPO" "$issue_number" || true

    log "✅ Completed #${issue_number} ('${safe_word}')"
  done <<< "$candidates_text"

  sleep "$MW_POLL_SECONDS"
done
