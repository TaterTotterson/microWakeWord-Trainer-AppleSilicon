from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from watcher_cleanup import (
    OWNER_FILE, PACKAGE_SUFFIXES, SHARED_SCRATCH, PublishedCatalog, cleanup_word,
    mark_training_word, pending_words, scratch_paths, sweep_published,
)


ROOT = Path(__file__).resolve().parents[1]

# A small catalog fixture exercises the production validator's hash-checking
# contract without downloading models or depending on another repository.
VALIDATOR = '''import hashlib
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
def bundle_model_fields(path):
    value = json.loads(path.read_text())
    for name, digest in value['files'].items():
        if hashlib.sha256(path.with_name(name).read_bytes()).hexdigest() != digest:
            return {}
    return {'bundle_path': str(path.relative_to(REPO_ROOT))}
if __name__ == '__main__':
    Path('wake_word_manifest.json').write_text('{}\\n')
'''


class WatcherCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        logger = patch("watcher_cleanup.log")
        logger.start()
        self.addCleanup(logger.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "catalog"
        self.origin = self.root / "origin.git"
        self.data = self.root / "data"
        self.support = self.root / "support"
        self.reports = self.root / "reports"
        self.data.mkdir()
        self.support.mkdir()
        self.git(self.root, "init", "--bare", str(self.origin))
        self.git(self.root, "init", "-b", "main", str(self.repo))
        self.git(self.repo, "config", "user.name", "Cleanup Test")
        self.git(self.repo, "config", "user.email", "cleanup-test@example.com")
        self.git(self.repo, "remote", "add", "origin", str(self.origin))
        (self.repo / "scripts").mkdir()
        (self.repo / "scripts/generate_wake_word_manifest.py").write_text(VALIDATOR)
        self.write_package("alpha")
        self.publish()
        self.kept = []
        for relative in (
            "personal_samples/recording.wav", "negative_samples/recording.wav",
            "captured_audio/recording.wav", "trim_history/original.wav",
            "shared_feature_banks/bank.npy", "negative_datasets/speech/data.npy",
            "audioset/background.wav", ".venv/bin/python",
            "wake_word_library/words/alpha/runs/v1/alpha.onnx", "trained_wake_words/alpha.onnx",
        ):
            path = self.data / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"keep")
            self.kept.append(path)

    @staticmethod
    def git(repo: Path, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE).decode()

    def write_package(self, word: str) -> None:
        directory = self.repo / "microWakeWordsV9"
        directory.mkdir(exist_ok=True)
        records = {}
        for suffix in PACKAGE_SUFFIXES[1:]:
            path = directory / f"{word}{suffix}"
            path.write_bytes(f"{word}{suffix}".encode())
            records[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (directory / f"{word}.wake-bundle.json").write_text(json.dumps({"files": records}))

    def publish(self) -> None:
        self.git(self.repo, "add", ".")
        self.git(self.repo, "commit", "-m", "Catalog update")
        self.git(self.repo, "push", "-u", "origin", "main")

    def catalog(self) -> PublishedCatalog:
        return PublishedCatalog(self.repo, "microWakeWordsV9", "main")

    def prepare_scratch(self, word="alpha", *, shared=True, owner=True) -> list[Path]:
        if owner:
            mark_training_word(self.data, word)
        paths = scratch_paths(self.data, word)
        if shared and not owner:
            paths.extend(self.data / name for name in SHARED_SCRATCH)
        for path in paths:
            if path.suffix == ".yaml":
                path.write_text("training config\n")
            else:
                path.mkdir(parents=True, exist_ok=True)
                (path / "scratch.bin").write_bytes(b"rebuildable")
                if path.name == word:
                    (path / f"{word}.quality.json").write_text('{"passed": true}')
        return paths

    def assert_kept(self) -> None:
        for path in self.kept:
            self.assertEqual(path.read_bytes(), b"keep", str(path))
        self.assertTrue((self.repo / "microWakeWordsV9/alpha.wake-bundle.json").is_file())

    def test_verified_publication_cleans_scratch_and_archives_reports(self) -> None:
        paths = self.prepare_scratch()
        catalog = self.catalog()
        self.assertEqual(cleanup_word(catalog, self.data, "alpha", self.reports), len(paths))
        self.assertTrue(all(not path.exists() for path in paths))
        self.assertFalse((self.data / OWNER_FILE).exists())
        report = self.reports / "alpha" / catalog.commit / "nanowakeword/output/cnn/alpha/alpha.quality.json"
        self.assertEqual(json.loads(report.read_text()), {"passed": True})
        self.assert_kept()

    def test_uncommitted_package_changes_prevent_deletion(self) -> None:
        paths = self.prepare_scratch()
        (self.repo / "microWakeWordsV9/alpha.tflite").write_bytes(b"unpublished")
        with self.assertRaises(ValueError):
            cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertTrue(all(path.exists() for path in paths))

    def test_unpushed_word_is_preserved(self) -> None:
        self.write_package("new_word")
        self.git(self.repo, "add", ".")
        self.git(self.repo, "commit", "-m", "Unpushed training")
        paths = self.prepare_scratch("new_word")
        with self.assertRaises(subprocess.CalledProcessError):
            cleanup_word(self.catalog(), self.data, "new_word", self.reports)
        self.assertTrue(all(path.exists() for path in paths))

    def test_corrupt_published_package_is_preserved(self) -> None:
        paths = self.prepare_scratch()
        (self.repo / "microWakeWordsV9/alpha.boundary.tflite").write_bytes(b"wrong hash")
        self.publish()
        with self.assertRaises(ValueError):
            cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertTrue(all(path.exists() for path in paths))

    def test_incomplete_published_package_is_preserved(self) -> None:
        paths = self.prepare_scratch()
        (self.repo / "microWakeWordsV9/alpha.nww.gru.onnx").unlink()
        self.publish()
        with self.assertRaises(subprocess.CalledProcessError):
            cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertTrue(all(path.exists() for path in paths))

    def test_other_words_shared_scratch_is_preserved(self) -> None:
        self.prepare_scratch(owner=False)
        mark_training_word(self.data, "unfinished")
        cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertTrue((self.data / "boundary_negative_features/scratch.bin").exists())
        self.assertEqual(json.loads((self.data / OWNER_FILE).read_text())["safe_word"], "unfinished")
        self.assert_kept()

    def test_unowned_shared_scratch_is_preserved(self) -> None:
        self.prepare_scratch(owner=False)
        cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertTrue((self.data / "generated_augmented_features/scratch.bin").exists())

    def test_changed_ui_word_preserves_shared_scratch(self) -> None:
        self.prepare_scratch()
        (self.data / "wake_word_library/active_word.json").write_text('{"safe_word":"another_word"}')
        cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertTrue((self.data / "generated_augmented_features/scratch.bin").exists())

    def test_another_words_generated_corpus_preserves_shared_scratch(self) -> None:
        self.prepare_scratch()
        (self.data / "generated_samples/.generation_manifest.json").write_text(
            '{"signature":{"phrase":"another word"}}'
        )
        cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertTrue((self.data / "generated_augmented_features/scratch.bin").exists())

    def test_reports_can_be_archived_without_copying_on_full_volume(self) -> None:
        self.prepare_scratch()
        with patch("watcher_cleanup.shutil.copyfile", side_effect=OSError("disk full")):
            cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertFalse((self.data / "nanowakeword/output/gru/alpha").exists())

    def test_symlink_leaf_is_removed_without_touching_target(self) -> None:
        self.prepare_scratch(owner=False)
        leaf = self.data / "openwakeword/output/alpha"
        import shutil
        shutil.rmtree(leaf)
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "private.wav").write_bytes(b"original")
        leaf.symlink_to(outside, target_is_directory=True)
        cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertFalse(leaf.is_symlink())
        self.assertEqual((outside / "private.wav").read_bytes(), b"original")

    def test_symlinked_parent_prevents_all_deletion(self) -> None:
        paths = self.prepare_scratch()
        output = self.data / "openwakeword/output"
        moved = self.data / "retained-output"
        output.rename(moved)
        output.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(ValueError):
            cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertTrue(all(path.exists() for path in paths))

    def test_internal_symlinks_do_not_move_or_delete_external_reports(self) -> None:
        self.prepare_scratch(owner=False)
        outside = self.root / "external-reports"
        outside.mkdir()
        original = outside / "alpha.calibration.json"
        original.write_text('{"original":true}')
        nested = self.data / "openwakeword/output/alpha/alpha"
        nested.symlink_to(outside, target_is_directory=True)
        cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertEqual(original.read_text(), '{"original":true}')

    def test_dry_run_does_not_remove_data_or_marker(self) -> None:
        paths = self.prepare_scratch()
        cleanup_word(self.catalog(), self.data, "alpha", self.reports, dry_run=True)
        self.assertTrue(all(path.exists() for path in paths))
        self.assertTrue((self.data / OWNER_FILE).exists())

    def test_archive_failure_prevents_all_deletion(self) -> None:
        paths = self.prepare_scratch()
        with patch("watcher_cleanup.archive_reports", side_effect=OSError("archive unavailable")):
            with self.assertRaises(OSError):
                cleanup_word(self.catalog(), self.data, "alpha", self.reports)
        self.assertTrue(all(path.exists() for path in paths))

    def test_startup_sweep_skips_pending_and_unpublished_words(self) -> None:
        self.write_package("pending")
        self.publish()
        alpha = self.prepare_scratch("alpha", owner=False)
        pending = self.prepare_scratch("pending", owner=False)
        unknown = self.prepare_scratch("unfinished", owner=False)
        sweep_published(self.catalog(), self.data, self.reports, {"pending"})
        self.assertTrue(all(not path.exists() for path in alpha[:5]))
        self.assertTrue(all(path.exists() for path in pending[:5] + unknown[:5]))
        self.assert_kept()

    def test_startup_sweep_recovers_interrupted_shared_cleanup(self) -> None:
        paths = self.prepare_scratch()
        import shutil
        for path in paths[:5]:
            shutil.rmtree(path)
        sweep_published(self.catalog(), self.data, self.reports, set())
        self.assertFalse((self.data / "boundary_negative_features").exists())
        self.assertFalse((self.data / OWNER_FILE).exists())
        self.assert_kept()

    def test_pending_requests_include_failed_and_duplicate_word_requests(self) -> None:
        issues = self.root / "issues.json"
        issues.write_text(json.dumps([
            {"title": "mww: Hey Tater", "labels": [{"name": "mww-added"}]},
            {"title": "mww: Hey Tater", "labels": [{"name": "mww-failed"}]},
            {"title": "mww: processing", "labels": [{"name": "mww-processing"}]},
            {"title": "mww: completed", "labels": [{"name": "mww-added"}]},
        ]))
        self.assertEqual(pending_words(issues, "mww-added"), {"hey_tater", "processing"})

    def test_potentially_truncated_queue_defers_backlog_cleanup(self) -> None:
        issues = self.root / "issues.json"
        issues.write_text(json.dumps([{"title": "mww: pending", "labels": []}] * 1000))
        with self.assertRaises(ValueError):
            pending_words(issues, "mww-added")

    def helper_command(self) -> list[str]:
        return ["python3", str(ROOT / "watcher_cleanup.py"), "word", "--word", "alpha",
                "--repo", str(self.repo), "--data-dir", str(self.data),
                "--support-dir", str(self.support), "--reports-dir", str(self.reports)]

    def test_running_ui_training_prevents_cleanup(self) -> None:
        paths = self.prepare_scratch()
        with (self.support / "training.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(self.helper_command(), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(all(path.exists() for path in paths))

    def test_remote_outage_prevents_cleanup(self) -> None:
        paths = self.prepare_scratch()
        self.git(self.repo, "remote", "set-url", "origin", str(self.root / "unavailable.git"))
        result = subprocess.run(self.helper_command(), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(all(path.exists() for path in paths))

    def test_invalid_word_cannot_escape_data_root(self) -> None:
        with self.assertRaises(ValueError):
            cleanup_word(self.catalog(), self.data, "../../personal_samples", self.reports)
        self.assert_kept()

    def run_watcher(self, *, worker_rc=0, publish=True, cleanup=True) -> subprocess.CompletedProcess[str]:
        word = "fresh"
        self.prepare_scratch(word, owner=False)
        producer = self.root / "produce.py"
        producer.write_text('''import hashlib, json
from pathlib import Path
directory = Path('microWakeWordsV9')
records = {}
for suffix in ('.esphome.json', '.tflite', '.oww.onnx', '.nww.cnn.onnx', '.nww.gru.onnx', '.boundary.tflite'):
    path = directory / ('fresh' + suffix)
    path.write_bytes(suffix.encode())
    records[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
(directory / 'fresh.wake-bundle.json').write_text(json.dumps({'files': records}))
''')
        worker = self.repo / "scripts/train_issue_wake_word.sh"
        worker.write_text(
            '#!/bin/bash\nset -e\n'
            'if [ -f microWakeWordsV9/fresh.wake-bundle.json ]; then exit 0; fi\n'
            'python3 "$FAKE_PRODUCER"\n'
            'python3 scripts/generate_wake_word_manifest.py\n'
            'git add .\ngit commit -m "Add wake word: fresh"\n'
            + ('git push origin HEAD:main\n' if publish else '')
            + f'exit {worker_rc}\n'
        )
        self.publish()
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        gh = fake_bin / "gh"
        gh.write_text('#!/bin/sh\n'
                      'if [ "$1" = auth ]; then exit 0; fi\n'
                      'if [ "$1" = issue ] && [ "$2" = list ]; then\n'
                      '  echo \'[{"number":99,"title":"mww: fresh","labels":[],"createdAt":"2026-01-01"}]\'\n'
                      '  exit 0\nfi\nexit 2\n')
        gh.chmod(0o755)
        env = os.environ.copy()
        env.update({
            "PATH": f"{fake_bin}:{env['PATH']}", "FAKE_PRODUCER": str(producer),
            "MW_ONCE": "1", "MW_DRY_RUN": "0", "MW_CLEANUP_AFTER_PUBLISH": "1" if cleanup else "0",
            "MW_CLEANUP_PUBLISHED_BACKLOG": "1", "MW_CATALOG_REPO_DIR": str(self.repo),
            "MW_WATCHER_STATE_DIR": str(self.root / "state"),
            "TATER_WAKE_TMP_ROOT": str(self.root / "tmp"),
            "TATER_WAKE_DATA_DIR": str(self.data), "TATER_WAKE_SUPPORT_DIR": str(self.support),
        })
        return subprocess.run(["/bin/bash", str(ROOT / "watcher.sh")], env=env,
                              cwd=ROOT, capture_output=True, text=True, timeout=30)

    def test_watcher_cleans_after_successful_publish(self) -> None:
        result = self.run_watcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Cleaned", result.stdout)
        self.assertFalse((self.data / "nanowakeword/output/gru/fresh").exists())
        self.assertFalse((self.data / "boundary_negative_features").exists())
        self.assert_kept()

    def test_watcher_reclaims_old_published_words_before_training(self) -> None:
        self.prepare_scratch("alpha", owner=False)
        result = self.run_watcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.data / "nanowakeword/output/gru/alpha").exists())
        self.assertLess(result.stdout.index("scratch paths for published word alpha"),
                        result.stdout.index("Processing #99"))
        self.assert_kept()

    def test_watcher_cleanup_can_be_disabled_for_debugging(self) -> None:
        result = self.run_watcher(cleanup=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.data / "nanowakeword/output/gru/fresh/scratch.bin").exists())
        self.assertTrue((self.data / "boundary_negative_features/scratch.bin").exists())

    def test_watcher_keeps_failed_training_workspace(self) -> None:
        result = self.run_watcher(worker_rc=1, publish=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.data / "nanowakeword/output/gru/fresh/scratch.bin").exists())
        self.assertTrue((self.data / "boundary_negative_features/scratch.bin").exists())

    def test_watcher_recovers_deferred_publish_before_cleanup(self) -> None:
        result = self.run_watcher(worker_rc=75, publish=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Recovered the published package", result.stdout)
        self.assertFalse((self.data / "nanowakeword/output/gru/fresh").exists())
        self.assertFalse((self.data / "boundary_negative_features").exists())

    def test_worker_success_without_remote_publication_does_not_clean(self) -> None:
        result = self.run_watcher(publish=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("cleanup was deferred", result.stdout)
        self.assertTrue((self.data / "nanowakeword/output/gru/fresh/scratch.bin").exists())
        self.assertTrue((self.data / "boundary_negative_features/scratch.bin").exists())


if __name__ == "__main__":
    unittest.main()
