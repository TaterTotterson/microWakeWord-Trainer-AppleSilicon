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
    def run_dry_watcher(self, *, retry_failed: bool = False) -> subprocess.CompletedProcess[str]:
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
        self.assertIn("microWakeWordsV7", output)
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


if __name__ == "__main__":
    unittest.main()
