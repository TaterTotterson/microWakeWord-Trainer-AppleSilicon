#!/usr/bin/env python3
"""Reclaim headless watcher scratch data only after verified catalog publication."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata


PACKAGE_SUFFIXES = (
    ".wake-bundle.json", ".esphome.json", ".tflite", ".oww.onnx",
    ".nww.cnn.onnx", ".nww.gru.onnx", ".boundary.tflite",
)
SHARED_SCRATCH = (
    "generated_samples", ".wake_word_samples.build", "generated_augmented_features",
    "generated_negative_samples", "generated_negative_features",
    "boundary_negative_samples", "boundary_negative_features",
    "personal_augmented_features", "reviewed_negative_features", "trained_models",
    "training_parameters.yaml", "boundary_training_parameters.yaml",
    "nanowakeword/output/cnn/_trainer_assets", "nanowakeword/output/gru/_trainer_assets",
)
OWNER_FILE = ".watcher-training-word.json"
REPORT_SUFFIXES = {".json", ".yaml", ".yml", ".md", ".log", ".txt"}


def log(message: str) -> None:
    print(f"[mw_watcher] {message}", flush=True)


def checked_word(word: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", word):
        raise ValueError(f"Invalid cleanup word: {word!r}")
    return word


def safe_slug(phrase: str) -> str:
    normalized = unicodedata.normalize("NFKC", phrase).strip().lower()
    word = re.sub(r"[^a-z0-9_]+", "", re.sub(r"\s+", "_", normalized)).strip("_")
    return word or f"wakeword_{hashlib.sha256(normalized.encode()).hexdigest()[:8]}"


def mark_training_word(data_dir: Path, word: str) -> None:
    checked_word(word)
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / OWNER_FILE
    if marker.is_symlink():
        raise ValueError("Training ownership marker must not be a symlink")
    temporary = data_dir / f"{OWNER_FILE}.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps({"safe_word": word}) + "\n", encoding="utf-8")
        temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)


def owns_shared_scratch(data_dir: Path, word: str) -> bool:
    marker = data_dir / OWNER_FILE
    if marker.is_symlink():
        return False
    try:
        if json.loads(marker.read_text(encoding="utf-8")).get("safe_word") != word:
            return False
        active = data_dir / "wake_word_library/active_word.json"
        if active.exists() and json.loads(active.read_text()).get("safe_word") != word:
            return False
        manifest = data_dir / "generated_samples/.generation_manifest.json"
        if manifest.exists():
            value = json.loads(manifest.read_text())
            phrase = value.get("signature", {}).get("phrase") or value.get("phrase")
            if not isinstance(phrase, str) or safe_slug(phrase) != word:
                return False
        return True
    except (OSError, ValueError, AttributeError):
        return False


class PublishedCatalog:
    def __init__(self, repo: Path, catalog_dir: str, branch: str):
        self.repo = repo.resolve()
        self.catalog_dir = PurePosixPath(catalog_dir)
        if self.catalog_dir.is_absolute() or not self.catalog_dir.parts or ".." in self.catalog_dir.parts:
            raise ValueError("Catalog directory must be a relative directory")
        # The claim heartbeat also fetches refs. Do not share FETCH_HEAD with it.
        ref = f"refs/remotes/origin/{branch}"
        self.git("fetch", "--quiet", "--no-tags", "--no-write-fetch-head", "origin",
                 f"refs/heads/{branch}:{ref}")
        self.commit = self.git("rev-parse", f"{ref}^{{commit}}").decode().strip()
        spec = importlib.util.spec_from_file_location(
            "watcher_catalog_validator", self.repo / "scripts/generate_wake_word_manifest.py"
        )
        if spec is None or spec.loader is None:
            raise ValueError("Catalog validator is unavailable")
        self.validator = importlib.util.module_from_spec(spec)
        # Importing the catalog validator must not dirty its Git checkout.
        sys.dont_write_bytecode = True
        spec.loader.exec_module(self.validator)

    def git(self, *args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE)

    def verify(self, word: str) -> None:
        """Require all seven local files to match a validated remote snapshot."""
        checked_word(word)
        with tempfile.TemporaryDirectory(prefix="mww-published-check-") as temporary:
            staging = Path(temporary).resolve()
            self.validator.REPO_ROOT = staging
            for suffix in PACKAGE_SUFFIXES:
                relative = self.catalog_dir / f"{word}{suffix}"
                published = self.git("show", f"{self.commit}:{relative}")
                local = self.repo / relative
                if local.is_symlink() or local.read_bytes() != published:
                    raise ValueError(f"Local package for {word} differs from GitHub; keeping its workspace")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(published)
            bundle = staging / self.catalog_dir / f"{word}.wake-bundle.json"
            if not self.validator.bundle_model_fields(bundle):
                raise ValueError(f"Published package for {word} failed validation; keeping its workspace")


def scratch_paths(data_dir: Path, word: str) -> list[Path]:
    checked_word(word)
    paths = [
        data_dir / "openwakeword/output" / word,
        data_dir / "openwakeword/publish-staging" / word,
        data_dir / "nanowakeword/output/cnn" / word,
        data_dir / "nanowakeword/output/gru" / word,
        data_dir / "nanowakeword/publish-staging" / word,
    ]
    if owns_shared_scratch(data_dir, word):
        paths.extend(data_dir / name for name in SHARED_SCRATCH)
    return paths


def check_path(path: Path, data_dir: Path, protected: tuple[Path, ...]) -> None:
    # Never follow a parent symlink out of the configured data root. A leaf
    # symlink can be removed as a link; its target must never be traversed.
    parent = path.parent.resolve()
    if not parent.is_relative_to(data_dir) or parent != path.parent:
        raise ValueError(f"Cleanup path has a symlinked parent or escapes the trainer data folder: {path}")
    effective = parent / path.name
    if effective == data_dir or any(p == effective or p.is_relative_to(effective) for p in protected):
        raise ValueError(f"Cleanup path contains protected data: {path}")


def archive_reports(path: Path, destination: Path) -> None:
    if path.is_symlink() or not path.exists():
        return
    reports = []
    if path.is_file():
        reports = [path] if path.suffix in REPORT_SUFFIXES else []
    elif path.is_dir():
        reports = [p for p in path.iterdir() if p.is_file() and p.suffix in REPORT_SUFFIXES]
        # OWW calibration and MWW calibration/configs live below their roots.
        reports.extend(path.glob(f"{path.name}/*.calibration.json"))
        for relative in (
            "wakeword/training_config.yaml",
            "wakeword/tflite_stream_state_internal_quant/detection_calibration.json",
            "wake_boundary/training_config.yaml",
            "wake_boundary/tflite_stream_state_internal_quant/boundary_calibration.json",
        ):
            candidate = path / relative
            if candidate.is_file():
                reports.append(candidate)
    for report in reports:
        if report.is_symlink() or report.parent.resolve() != report.parent:
            continue
        target = destination / report.relative_to(path) if path.is_dir() else destination
        target.parent.mkdir(parents=True, exist_ok=True)
        # Rename on the usual shared volume, avoiding a second copy when the
        # training disk is already full. Cross-volume moves copy before unlink.
        shutil.move(str(report), str(target))


def cleanup_word(
    catalog: PublishedCatalog, data_dir: Path, word: str, reports_dir: Path,
    *, dry_run: bool = False,
) -> int:
    data_dir = data_dir.resolve()
    reports_dir = reports_dir.resolve()
    catalog.verify(word)
    paths = scratch_paths(data_dir, word)
    protected = (catalog.repo, reports_dir, data_dir / "trained_wake_words", data_dir / "wake_word_library")
    for path in paths:
        check_path(path, data_dir, protected)
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if dry_run:
        for path in existing:
            log(f"DRY RUN: would remove published-word scratch: {path}")
        return len(existing)

    # Preserve compact diagnostics before deleting anything. If archiving fails,
    # leave all scratch data intact so the cleanup can be retried later.
    for path in existing:
        archive_reports(path, reports_dir / word / catalog.commit / path.relative_to(data_dir))
    before = shutil.disk_usage(data_dir).free
    for path in existing:
        if not path.exists() and not path.is_symlink():
            # Standalone config files were moved into the diagnostic archive.
            continue
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    if owns_shared_scratch(data_dir, word):
        (data_dir / OWNER_FILE).unlink()
    freed = max(0, shutil.disk_usage(data_dir).free - before)
    log(f"Cleaned {len(existing)} scratch paths for published word {word}; "
        f"{freed / 1024**3:.2f} GiB reclaimed. Final models, JSONs, datasets, recordings and logs kept.")
    return len(existing)


def pending_words(issues_path: Path, done_label: str) -> set[str]:
    issues = json.loads(issues_path.read_text(encoding="utf-8"))
    if not isinstance(issues, list):
        raise ValueError("Open issue list must be an array")
    if len(issues) >= 1000:
        raise ValueError("Open issue list may be truncated; keeping the backlog until all pending words are known")
    words = set()
    for issue in issues:
        labels = {label.get("name") for label in issue.get("labels", [])}
        match = re.match(r"^\s*mww:\s*(.+?)\s*$", issue.get("title", ""), flags=re.I)
        if match and done_label not in labels:
            words.add(safe_slug(match.group(1)))
    return words


def sweep_published(
    catalog: PublishedCatalog, data_dir: Path, reports_dir: Path, pending: set[str],
    *, dry_run: bool = False,
) -> None:
    data_dir = data_dir.resolve()
    words = set()
    for relative in (
        "openwakeword/output", "openwakeword/publish-staging", "nanowakeword/output/cnn",
        "nanowakeword/output/gru", "nanowakeword/publish-staging",
    ):
        directory = data_dir / relative
        check_path(directory / "placeholder", data_dir.resolve(), (catalog.repo, reports_dir.resolve()))
        if directory.is_dir():
            words.update(p.name for p in directory.iterdir()
                         if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", p.name) and p.is_dir())
    # Retry an interrupted cleanup even if it already removed every named
    # OWW/NWW directory before reaching the shared MWW scratch paths.
    marker = data_dir / OWNER_FILE
    if not marker.is_symlink():
        try:
            owner = json.loads(marker.read_text()).get("safe_word")
            if isinstance(owner, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]*", owner):
                words.add(owner)
        except (OSError, ValueError, AttributeError):
            pass
    for word in sorted(words - pending):
        try:
            cleanup_word(catalog, data_dir, word, reports_dir, dry_run=dry_run)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            log(f"Keeping scratch for {word}: {error}")


@contextmanager
def training_lock(support_dir: Path):
    filename = os.environ.get("WAKEWORD_TRAINER_TRAINING_LOCK_FILE")
    path = Path(filename) if filename else support_dir / "training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("mark", "word", "sweep"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--word")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--catalog-dir", default="microWakeWordsV9")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--issues", type=Path)
    parser.add_argument("--done-label", default="mww-added")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        with training_lock(args.support_dir):
            if args.action == "mark":
                if not args.dry_run:
                    mark_training_word(args.data_dir.resolve(), args.word or "")
            else:
                if args.repo is None or args.reports_dir is None:
                    raise ValueError("Cleanup requires a catalog checkout and reports folder")
                catalog = PublishedCatalog(args.repo, args.catalog_dir, args.branch)
                if args.action == "word":
                    cleanup_word(catalog, args.data_dir, args.word or "", args.reports_dir, dry_run=args.dry_run)
                else:
                    if args.issues is None:
                        raise ValueError("Backlog cleanup requires the open issue list")
                    pending = pending_words(args.issues, args.done_label)
                    sweep_published(catalog, args.data_dir, args.reports_dir, pending, dry_run=args.dry_run)
    except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as error:
        log(f"Cleanup deferred; remaining training data kept: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
