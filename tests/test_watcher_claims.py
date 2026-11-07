from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLAIM_TOOL = ROOT / "watcher_claims.sh"


class WatcherClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp = Path(self.temp_dir.name)
        self.origin = self.temp / "origin.git"
        seed = self.temp / "seed"

        self.git("init", "--bare", str(self.origin), cwd=self.temp)
        self.git("init", str(seed), cwd=self.temp)
        self.git("checkout", "-b", "main", cwd=seed)
        self.configure_git(seed)
        (seed / "README.md").write_text("claim fixture\n", encoding="utf-8")
        self.git("add", "README.md", cwd=seed)
        self.git("commit", "-m", "Initial catalog", cwd=seed)
        self.git("remote", "add", "origin", str(self.origin), cwd=seed)
        self.git("push", "-u", "origin", "main", cwd=seed)
        self.git("symbolic-ref", "HEAD", "refs/heads/main", cwd=self.origin)

        self.watcher_a = self.temp / "watcher-a"
        self.watcher_b = self.temp / "watcher-b"
        self.git("clone", str(self.origin), str(self.watcher_a), cwd=self.temp)
        self.git("clone", str(self.origin), str(self.watcher_b), cwd=self.temp)
        self.configure_git(self.watcher_a)
        self.configure_git(self.watcher_b)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def claim(
        self,
        repo: Path,
        action: str,
        issue: int,
        watcher_id: str,
        *,
        now: int,
        ttl: int = 86_400,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["MW_CLAIM_NOW_EPOCH"] = str(now)
        return subprocess.run(
            [
                "/bin/bash",
                str(CLAIM_TOOL),
                action,
                str(repo),
                str(issue),
                watcher_id,
                str(ttl),
            ],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def test_two_watchers_claim_different_jobs_without_collision(self) -> None:
        first = self.claim(self.watcher_a, "acquire", 41, "mac-mini", now=1_000)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("acquired\t41\tmac-mini\t87400", first.stdout)

        collision = self.claim(
            self.watcher_b, "acquire", 41, "mac-studio", now=1_001
        )
        self.assertEqual(collision.returncode, 76, collision.stderr)
        self.assertIn("busy\t41\tmac-mini\t87400", collision.stdout)

        next_job = self.claim(
            self.watcher_b, "acquire", 42, "mac-studio", now=1_001
        )
        self.assertEqual(next_job.returncode, 0, next_job.stderr)
        self.assertIn("acquired\t42\tmac-studio", next_job.stdout)

        not_owner = self.claim(
            self.watcher_b, "release", 41, "mac-studio", now=1_002
        )
        self.assertEqual(not_owner.returncode, 0, not_owner.stderr)
        self.assertIn("not-owner\t41\tmac-mini", not_owner.stdout)

        released = self.claim(
            self.watcher_a, "release", 41, "mac-mini", now=1_003
        )
        self.assertEqual(released.returncode, 0, released.stderr)
        reclaimed = self.claim(
            self.watcher_b, "acquire", 41, "mac-studio", now=1_004
        )
        self.assertEqual(reclaimed.returncode, 0, reclaimed.stderr)
        self.assertIn("acquired\t41\tmac-studio", reclaimed.stdout)

    def test_restart_renews_lease_and_expired_claim_can_be_recovered(self) -> None:
        acquired = self.claim(
            self.watcher_a, "acquire", 51, "mac-mini", now=1_000, ttl=100
        )
        self.assertEqual(acquired.returncode, 0, acquired.stderr)

        restarted = self.claim(
            self.watcher_a, "acquire", 51, "mac-mini", now=1_090, ttl=100
        )
        self.assertEqual(restarted.returncode, 0, restarted.stderr)
        self.assertIn("renewed\t51\tmac-mini\t1190", restarted.stdout)

        still_active = self.claim(
            self.watcher_b, "acquire", 51, "mac-studio", now=1_101, ttl=100
        )
        self.assertEqual(still_active.returncode, 76, still_active.stderr)
        self.assertIn("busy\t51\tmac-mini\t1190", still_active.stdout)

        takeover = self.claim(
            self.watcher_b, "acquire", 51, "mac-studio", now=1_191, ttl=100
        )
        self.assertEqual(takeover.returncode, 0, takeover.stderr)
        self.assertIn("acquired\t51\tmac-studio\t1291", takeover.stdout)

        old_owner_renewal = self.claim(
            self.watcher_a, "renew", 51, "mac-mini", now=1_192, ttl=100
        )
        self.assertEqual(old_owner_renewal.returncode, 76, old_owner_renewal.stderr)
        self.assertIn("lost\t51\tmac-studio\t1291", old_owner_renewal.stdout)

    @staticmethod
    def git(*args: str, cwd: Path) -> None:
        subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )

    @classmethod
    def configure_git(cls, repo: Path) -> None:
        cls.git("config", "user.name", "Watcher Test", cwd=repo)
        cls.git("config", "user.email", "watcher-test@example.com", cwd=repo)


if __name__ == "__main__":
    unittest.main()
