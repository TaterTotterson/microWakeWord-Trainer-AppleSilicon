from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WATCHER = ROOT / "watcher.sh"


class WatcherTests(unittest.TestCase):
    def run_dry_watcher(
        self,
        *,
        retry_failed: bool = False,
        recovered_failed: bool = False,
        recovered_boundary: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        issues = [
            {
                "number": 15,
                "title": "mww: newest request",
                "createdAt": "2026-01-05T00:00:00Z",
                "labels": [],
            },
            {
                "number": 11,
                "title": "mww: oldest request",
                "createdAt": "2026-01-01T00:00:00Z",
                "labels": [{"name": "wake-word-request"}],
            },
            {
                "number": 12,
                "title": "mww: resume me",
                "createdAt": "2026-01-02T00:00:00Z",
                "labels": [{"name": "mww-processing"}],
            },
            {
                "number": 13,
                "title": "mww: failed before",
                "createdAt": "2026-01-03T00:00:00Z",
                "labels": [{"name": "mww-failed"}],
            },
            {
                "number": 14,
                "title": "mww: already done",
                "createdAt": "2026-01-04T00:00:00Z",
                "labels": [{"name": "mww-added"}],
            },
            {
                "number": 16,
                "title": "regular issue",
                "createdAt": "2026-01-06T00:00:00Z",
                "labels": [],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            fake_bin = temp / "bin"
            fake_bin.mkdir()
            issues_path = temp / "issues.json"
            issues_path.write_text(json.dumps(issues), encoding="utf-8")

            catalog_repo = temp / "catalog"
            if recovered_failed:
                catalog_dir = catalog_repo / "microWakeWordsV9"
                catalog_dir.mkdir(parents=True)
                base = catalog_dir / "failed_before"
                for suffix in (
                    ".wake-bundle.json",
                    ".esphome.json",
                    ".tflite",
                    ".oww.onnx",
                    ".nww.cnn.onnx",
                    ".nww.gru.onnx",
                    ".boundary.tflite",
                ):
                    if suffix == ".boundary.tflite" and not recovered_boundary:
                        continue
                    Path(f"{base}{suffix}").write_text("complete", encoding="utf-8")

            gh = fake_bin / "gh"
            gh.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "issue" ] && [ "$2" = "list" ]; then\n'
                '  exec /bin/cat "$FAKE_GH_ISSUES"\n'
                "fi\n"
                'echo "unexpected gh command: $*" >&2\n'
                "exit 2\n",
                encoding="utf-8",
            )
            gh.chmod(0o755)

            state = temp / "state"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "FAKE_GH_ISSUES": str(issues_path),
                    "MW_ONCE": "1",
                    "MW_DRY_RUN": "1",
                    "MW_RETRY_FAILED": "1" if retry_failed else "0",
                    "MW_WATCHER_STATE_DIR": str(state),
                    "MW_CATALOG_REPO_DIR": str(catalog_repo),
                    "TATER_WAKE_TMP_ROOT": str(temp / "runtime"),
                }
            )
            result = subprocess.run(
                ["/bin/bash", str(WATCHER)],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertFalse((state / "lock").exists())
            return result

    def test_dry_run_reclaims_processing_and_orders_oldest_first(self) -> None:
        result = self.run_dry_watcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout
        self.assertIn("TaterTotterson/Tater-Wake-Words", output)
        self.assertIn("microWakeWordsV9", output)
        self.assertIn("#11 'oldest request'", output)
        self.assertIn("#12 'resume me'", output)
        self.assertIn("#15 'newest request'", output)
        self.assertLess(output.index("#11"), output.index("#12"))
        self.assertLess(output.index("#12"), output.index("#15"))
        self.assertNotIn("#13 'failed before'", output)
        self.assertNotIn("#14 'already done'", output)
        self.assertNotIn("#16", output)

    def test_failed_requests_can_be_explicitly_retried(self) -> None:
        result = self.run_dry_watcher(retry_failed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("#13 'failed before'", result.stdout)

    def test_complete_local_package_reclaims_failed_request(self) -> None:
        result = self.run_dry_watcher(recovered_failed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("#13 'failed before'", result.stdout)

    def test_package_without_boundary_does_not_reclaim_failed_request(self) -> None:
        result = self.run_dry_watcher(
            recovered_failed=True,
            recovered_boundary=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("#13 'failed before'", result.stdout)

    def test_recovery_validates_absolute_paths_and_finalizes_deferred_pushes(self) -> None:
        watcher = WATCHER.read_text(encoding="utf-8")

        self.assertIn(
            "bundle_model_fields(Path(sys.argv[1]).resolve())",
            watcher,
        )
        self.assertIn("if [[ $train_rc -eq 75 ]]", watcher)
        self.assertIn("sync_catalog_checkout && catalog_package_is_complete", watcher)
        self.assertGreaterEqual(watcher.count("run_issue_worker"), 3)
        self.assertIn('export MW_JOB_LOG="$log_file"', watcher)
        self.assertIn('MW_LOG_DIR="${MW_WATCHER_LOG_DIR:', watcher)

    def test_diverged_catalog_regenerates_manifest_and_pushes(self) -> None:
        self.check_diverged_catalog()

    def test_duplicate_word_restores_whole_validated_published_package(self) -> None:
        self.check_diverged_catalog(duplicate_word=True)

    def test_duplicate_new_word_additions_keep_published_package(self) -> None:
        self.check_diverged_catalog(duplicate_word=True, new_word=True)

    def test_interrupted_duplicate_word_merge_recovers(self) -> None:
        self.check_diverged_catalog(duplicate_word=True, interrupted_merge=True)

    def test_invalid_published_package_preserves_local_commits(self) -> None:
        self.check_diverged_catalog(duplicate_word=True, invalid_package=True)

    def test_incomplete_published_package_preserves_local_commits(self) -> None:
        self.check_diverged_catalog(duplicate_word=True, missing_artifact=True)

    def test_source_conflict_is_not_automatically_resolved(self) -> None:
        self.check_diverged_catalog(duplicate_word=True, source_conflict=True)

    def test_manifest_failure_does_not_commit_or_push_merge(self) -> None:
        self.check_diverged_catalog(generator_failure=True)

    def check_diverged_catalog(
        self, *, duplicate_word=False, interrupted_merge=False,
        invalid_package=False, source_conflict=False, generator_failure=False,
        new_word=False, missing_artifact=False,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            origin = temp / "origin.git"
            seed = temp / "seed"
            catalog = temp / "catalog"
            updater = temp / "updater"

            self.git("init", "--bare", str(origin), cwd=temp)
            self.git("init", str(seed), cwd=temp)
            self.git("checkout", "-b", "main", cwd=seed)
            self.configure_git(seed)

            (seed / "microWakeWordsV9").mkdir()
            (seed / "microWakeWordsV9" / "base.marker").write_text(
                "base\n", encoding="utf-8"
            )
            (seed / "scripts").mkdir()
            generator = seed / "scripts" / "generate_wake_word_manifest.py"
            generator.write_text(
                "import json\n"
                "import hashlib\n"
                "from pathlib import Path\n"
                "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
                # Exercise the catalog validator's contract with real hashes.
                "def bundle_model_fields(path):\n"
                "    bundle = json.loads(path.read_text())\n"
                "    for name, digest in bundle['files'].items():\n"
                "        artifact = path.with_name(name)\n"
                "        if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:\n"
                "            return {}\n"
                "    return {'bundle_path': str(path.relative_to(REPO_ROOT))}\n"
                "if __name__ == '__main__':\n"
                "    if Path('fail_manifest').exists():\n"
                "        raise SystemExit('manifest generation failed')\n"
                "    names = sorted(p.stem for p in Path('microWakeWordsV9').glob('*.marker'))\n"
                "    Path('wake_word_manifest.json').write_text(json.dumps(names) + '\\n')\n",
                encoding="utf-8",
            )
            trainer = seed / "scripts" / "train_issue_wake_word.sh"
            trainer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            trainer.chmod(0o755)
            if duplicate_word and not new_word:
                self.write_word_package(seed)
            self.run_generator(seed)
            self.git("add", ".", cwd=seed)
            self.git("commit", "-m", "Initial catalog", cwd=seed)
            self.git("remote", "add", "origin", str(origin), cwd=seed)
            self.git("push", "-u", "origin", "main", cwd=seed)
            self.git("symbolic-ref", "HEAD", "refs/heads/main", cwd=origin)

            self.git("clone", str(origin), str(catalog), cwd=temp)
            self.configure_git(catalog)
            if duplicate_word:
                # Only the boundary model overlaps the remote model changes.
                # The local-only OWW/ESPHome changes must not leak into the
                # remote package selected during conflict recovery.
                self.write_word_package(catalog, boundary="local", oww="local", esphome="local")
            if source_conflict:
                (catalog / "scripts/train_issue_wake_word.sh").write_text("local source\n")
            (catalog / "microWakeWordsV9" / "bart.marker").write_text(
                "bart\n", encoding="utf-8"
            )
            self.run_generator(catalog)
            if generator_failure:
                (catalog / "fail_manifest").write_text("fail after merge\n")
            self.git("add", ".", cwd=catalog)
            self.git("commit", "-m", "Add wake word: bart", cwd=catalog)
            local_head = self.git_output("rev-parse", "HEAD", cwd=catalog).strip()

            self.git("clone", str(origin), str(updater), cwd=temp)
            self.configure_git(updater)
            if duplicate_word:
                self.write_word_package(updater, boundary="published")
                if invalid_package:
                    (updater / "microWakeWordsV9/vision.boundary.tflite").write_bytes(b"wrong hash")
                if missing_artifact:
                    (updater / "microWakeWordsV9/vision.nww.gru.onnx").unlink()
            if source_conflict:
                (updater / "scripts/train_issue_wake_word.sh").write_text("remote source\n")
            (updater / "microWakeWordsV9" / "santa.marker").write_text(
                "santa\n", encoding="utf-8"
            )
            self.run_generator(updater)
            self.git("add", ".", cwd=updater)
            self.git("commit", "-m", "Add wake word: santa", cwd=updater)
            self.git("push", "origin", "main", cwd=updater)
            published_head = self.git_output("rev-parse", "HEAD", cwd=updater).strip()
            if interrupted_merge:
                self.git("fetch", "origin", "main", cwd=catalog)
                merge = subprocess.run(
                    ["git", "merge", "--no-commit", "origin/main"],
                    cwd=catalog, capture_output=True,
                )
                self.assertNotEqual(merge.returncode, 0)

            fake_bin = temp / "bin"
            fake_bin.mkdir()
            issues_path = temp / "issues.json"
            issues_path.write_text(
                json.dumps(
                    [
                        {
                            "number": 99,
                            "title": "mww: next request",
                            "createdAt": "2026-01-01T00:00:00Z",
                            "labels": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            gh = fake_bin / "gh"
            gh.write_text(
                "#!/bin/sh\n"
                'if [ "$1" = "auth" ] && [ "$2" = "setup-git" ]; then exit 0; fi\n'
                'if [ "$1" = "issue" ] && [ "$2" = "list" ]; then\n'
                '  exec /bin/cat "$FAKE_GH_ISSUES"\n'
                "fi\n"
                'echo "unexpected gh command: $*" >&2\n'
                "exit 2\n",
                encoding="utf-8",
            )
            gh.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "FAKE_GH_ISSUES": str(issues_path),
                    "MW_ONCE": "1",
                    "MW_DRY_RUN": "0",
                    "MW_CATALOG_REPO_DIR": str(catalog),
                    "MW_WATCHER_STATE_DIR": str(temp / "state"),
                    "TATER_WAKE_TMP_ROOT": str(temp / "runtime"),
                    "TATER_WAKE_DATA_DIR": str(temp / "trainer-data"),
                    "TATER_WAKE_SUPPORT_DIR": str(temp / "trainer-support"),
                }
            )
            result = subprocess.run(
                ["/bin/bash", str(WATCHER)],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            blocked = invalid_package or source_conflict or generator_failure or missing_artifact
            if blocked:
                self.assertIn("deferring the remaining requests", result.stdout)
                self.assertEqual(self.git_output("rev-parse", "HEAD", cwd=catalog).strip(), local_head)
                self.assertEqual(self.git_output("status", "--porcelain", cwd=catalog), "")
                remote_head = self.git_output("rev-parse", "main", cwd=origin).strip()
                self.assertEqual(remote_head, published_head)
                if invalid_package:
                    self.assertIn("failed validation", result.stderr)
                if source_conflict:
                    self.assertIn("non-generated conflict", result.stderr)
                if generator_failure:
                    self.assertIn("manifest generation failed", result.stderr)
                return

            self.assertIn("Resolving the generated wake-word manifest", result.stdout, result.stderr)
            if duplicate_word:
                self.assertIn("Keeping validated published package for vision", result.stdout)
                for original in (updater / "microWakeWordsV9").glob("vision.*"):
                    self.assertEqual(
                        original.read_bytes(),
                        (catalog / "microWakeWordsV9" / original.name).read_bytes(),
                        original.name,
                    )
                # The alternative trained locally is retained in Git history.
                self.git("merge-base", "--is-ancestor", local_head, "HEAD", cwd=catalog)
                preserved = self.git_output(
                    "show", f"{local_head}:microWakeWordsV9/vision.esphome.json", cwd=catalog
                )
                self.assertIn("local", preserved)
                self.assertEqual(self.git_output("status", "--porcelain", cwd=catalog), "")

            self.git("fetch", "origin", "main", cwd=seed)
            manifest = self.git_output(
                "show", "origin/main:wake_word_manifest.json", cwd=seed
            )
            self.assertEqual(json.loads(manifest), ["bart", "base", "santa"])

    @staticmethod
    def write_word_package(repo: Path, *, boundary="base", oww="base", esphome="base") -> None:
        directory = repo / "microWakeWordsV9"
        artifacts = {
            "vision.tflite": b"\0micro",
            "vision.oww.onnx": b"\0" + oww.encode(),
            "vision.nww.cnn.onnx": b"\0cnn",
            "vision.nww.gru.onnx": b"\0gru",
            "vision.boundary.tflite": b"\0" + boundary.encode(),
            "vision.esphome.json": json.dumps({"version": esphome}).encode(),
        }
        for name, content in artifacts.items():
            (directory / name).write_bytes(content)
        (directory / "vision.wake-bundle.json").write_text(json.dumps({
            "files": {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()}
        }, indent=2) + "\n")

    @staticmethod
    def git(*args: str, cwd: Path) -> None:
        subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )

    @staticmethod
    def git_output(*args: str, cwd: Path) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout

    @classmethod
    def configure_git(cls, repo: Path) -> None:
        cls.git("config", "user.name", "Watcher Test", cwd=repo)
        cls.git("config", "user.email", "watcher-test@example.com", cwd=repo)

    @staticmethod
    def run_generator(repo: Path) -> None:
        subprocess.run(
            ["python3", "scripts/generate_wake_word_manifest.py"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
