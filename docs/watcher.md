# GitHub issue watcher

[Back to the README](../README.md)

Run the commands below from the trainer repository root.

`watcher.sh` is the headless automation path for the dedicated training Mac. It
polls open `mww:` issues in
[`TaterTotterson/Tater-Wake-Words`](https://github.com/TaterTotterson/Tater-Wake-Words),
trains MWW, OWW, NanoWakeWord CNN, NanoWakeWord GRU, and the wake-boundary
model, commits the seven V9 artifacts plus the catalog manifest, comments on
the issue, and closes it. V9 uses one schema-v4 `.wake-bundle.json` in Tater,
including the boundary-model metadata, and keeps a separate `.esphome.json`
for ESPHome microWakeWord. Existing V8 packages remain unchanged.

The watcher does not depend on a continuously connected GitHub Actions runner.
Local training continues through a temporary network outage. Failed pushes are
kept in its dedicated catalog checkout and retried after the connection returns.
If GitHub advances meanwhile, the watcher merges both additions and regenerates
the derived catalog manifest. If duplicate requests produce conflicting models
for the same word, recovery keeps the complete, hash-validated package already
published in the catalog. It restores all seven matching artifacts together;
the local alternative remains recoverable in the merge's parent history.
Incomplete or invalid published packages and conflicts outside the known word
artifacts still stop recovery without discarding local work.
Issues left with `mww-processing`, or labeled
`mww-failed` despite having a complete local package, are reclaimed after a
restart without retraining.

Multiple Macs can run the watcher against the same issue queue. Before training,
each watcher atomically creates a temporary `mww-watcher-claim-*` Git tag in the
catalog repository. A Mac that loses the claim skips that issue and continues to
the next one. The claim is renewed while training and deleted when the job exits;
the same Mac immediately renews its claim after a restart. An abandoned claim
expires after 24 hours by default so another Mac can recover it. These
coordination tags are not release/version tags and do not trigger a workflow or
trainer release build.

Authenticate once with GitHub CLI:

```bash
gh auth login
```

Alternatively, create a local `.env` file containing a token with issue and
repository-content write access. `.env` is ignored by Git:

```bash
MW_GITHUB_TOKEN=github_pat_replace_me
```

Confirm the pending queue without changing anything:

```bash
MW_ONCE=1 MW_DRY_RUN=1 ./watcher.sh
```

Run it continuously in the current terminal:

```bash
./watcher.sh
```

For the training Mac, install it as a launchd service so it starts at login,
survives terminal disconnects, and restarts if the watcher exits:

```bash
./install_watcher_launchd.sh
```

The service writes to `~/Library/Logs/TaterWakeWordWatcher.log`. Check it with:

```bash
tail -f ~/Library/Logs/TaterWakeWordWatcher.log
```

Each request also keeps its complete training output in
`$TATER_WAKE_TMP_ROOT/watcher/logs/issue-<number>-<wake-word>.log` (normally
beside the trainer in `tater-wake-tmp`). If two watchers finish together, a
rejected catalog push is recovered by merging both additions and regenerating
the catalog manifest; it is not reported as failed model training.

After each successful job, the watcher fetches the catalog branch and checks
that all seven local package files match a complete, hash-validated package on
GitHub. Only then does it remove that word's OWW/NWW generated audio, features,
checkpoints, and redundant publication staging. Generated MWW audio, augmented
features, boundary examples/features, and checkpoints are removed only when
the watcher recorded that this word owns the shared training workspace.
Failed training, unpushed packages, mismatched files, and unverifiable GitHub
access keep their scratch data for recovery. Cleanup also respects the desktop
trainer's active training lock.

On startup, the watcher also checks old per-word workspaces against GitHub to
reclaim completed jobs left by older versions. Words with unfinished open
requests (including failed and processing requests) are skipped. Shared
datasets, the OWW/NWW feature bank, environments, personal recordings, reviewed
negatives, captured audio, saved library versions, the catalog checkout, and
final exports in `tater-wake-tmp/outputs/` are retained. Compact configs and
quality/calibration reports are archived beneath
`$MW_WATCHER_LOG_DIR/completed/<word>/<catalog-commit>/`; the complete issue logs
remain in the usual logs folder. This cleanup applies to the headless watcher;
desktop training continues to retain its per-word library and caches.

Remove the service with `./install_watcher_launchd.sh --uninstall`.

Useful overrides include:

- `MW_POLL_SECONDS=60` — issue polling interval.
- `MW_CATALOG_REPO_DIR=/path/to/checkout` — dedicated catalog checkout.
- `MW_CATALOG_DIR=microWakeWordsV9` — destination schema-v4 catalog folder.
- `TATER_WAKE_TMP_ROOT=/path/to/temp` — temp files, caches, and watcher state.
- `TATER_WAKE_DATA_DIR=/path/to/data` — persistent trainer datasets.
- `MW_WATCHER_ID=mac-studio` — optional readable identity; each Mac receives a
  stable generated identity automatically when this is omitted.
- `MW_CLAIM_TTL_SECONDS=86400` — abandoned cross-Mac claim expiry (24 hours by
  default; active training renews it continuously).
- `MW_CLAIM_REFRESH_SECONDS=300` — active-claim renewal interval.
- `MW_RETRY_FAILED=1` — include issues labeled `mww-failed`; leave this disabled
  for the continuous service to avoid repeatedly training a permanently broken
  request.
- `MW_CLEANUP_AFTER_PUBLISH=0` — keep watcher training scratch for debugging;
  cleanup is enabled by default.
- `MW_CLEANUP_PUBLISHED_BACKLOG=0` — disable the startup pass over old published
  words while retaining cleanup after each new successful job.

Run only one watcher per training Mac. Disable any competing automatic
self-hosted Actions training trigger on that Mac before starting the watcher.
Other Macs can run their own watchers using the shared claim mechanism above.
