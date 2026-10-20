from __future__ import annotations

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
        self, *, retry_failed: bool = False, recovered_failed: bool = False
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
                catalog_dir = catalog_repo / "microWakeWordsV8"
                catalog_dir.mkdir(parents=True)
                base = catalog_dir / "failed_before"
                for suffix in (
                    ".wake-bundle.json",
                    ".esphome.json",
                    ".tflite",
                    ".oww.onnx",
                    ".nww.cnn.onnx",
                    ".nww.gru.onnx",
                ):
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
        self.assertIn("microWakeWordsV8", output)
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

    def test_diverged_catalog_regenerates_manifest_and_pushes(self) -> None:
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

            (seed / "microWakeWordsV8").mkdir()
            (seed / "microWakeWordsV8" / "base.marker").write_text(
                "base\n", encoding="utf-8"
            )
            (seed / "scripts").mkdir()
            generator = seed / "scripts" / "generate_wake_word_manifest.py"
            generator.write_text(
                "import json\n"
                "from pathlib import Path\n"
                "names = sorted(p.stem for p in Path('microWakeWordsV8').glob('*.marker'))\n"
                "Path('wake_word_manifest.json').write_text(json.dumps(names) + '\\n')\n",
                encoding="utf-8",
            )
            trainer = seed / "scripts" / "train_issue_wake_word.sh"
            trainer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            trainer.chmod(0o755)
            self.run_generator(seed)
            self.git("add", ".", cwd=seed)
            self.git("commit", "-m", "Initial catalog", cwd=seed)
            self.git("remote", "add", "origin", str(origin), cwd=seed)
            self.git("push", "-u", "origin", "main", cwd=seed)
            self.git("symbolic-ref", "HEAD", "refs/heads/main", cwd=origin)

            self.git("clone", str(origin), str(catalog), cwd=temp)
            self.configure_git(catalog)
            (catalog / "microWakeWordsV8" / "bart.marker").write_text(
                "bart\n", encoding="utf-8"
            )
            self.run_generator(catalog)
            self.git("add", ".", cwd=catalog)
            self.git("commit", "-m", "Add wake word: bart", cwd=catalog)

            self.git("clone", str(origin), str(updater), cwd=temp)
            self.configure_git(updater)
            (updater / "microWakeWordsV8" / "santa.marker").write_text(
                "santa\n", encoding="utf-8"
            )
            self.run_generator(updater)
            self.git("add", ".", cwd=updater)
            self.git("commit", "-m", "Add wake word: santa", cwd=updater)
            self.git("push", "origin", "main", cwd=updater)

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
                    "MW_CATALOG_REPO_DIR": str(catalog),
                    "MW_WATCHER_STATE_DIR": str(temp / "state"),
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
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Resolving the generated wake-word manifest", result.stdout)

            self.git("fetch", "origin", "main", cwd=seed)
            manifest = self.git_output(
                "show", "origin/main:wake_word_manifest.json", cwd=seed
            )
            self.assertEqual(json.loads(manifest), ["bart", "base", "santa"])

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
