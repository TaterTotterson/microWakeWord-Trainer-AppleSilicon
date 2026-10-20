from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trainer_server as trainer


class _CompletedTrainingProcess:
    def __init__(self):
        self.pid = 6543
        self.returncode = 0
        self.stdout = io.StringIO("worker started\n")

    def wait(self, timeout=None):
        return self.returncode


class TrainingStageFailureTests(unittest.TestCase):
    def test_openwakeword_quality_rejection_is_logged_and_returns_stage_failure(self):
        original_process = trainer.TRAINING_PROCESS
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                staging = root / "staging"
                staging.mkdir()
                log_path = root / "recorder_training.log"
                process = _CompletedTrainingProcess()
                with (
                    patch.object(
                        trainer,
                        "prepare_openwakeword_stage",
                        return_value=(["oww-trainer"], root, {}, staging),
                    ),
                    patch.object(trainer.subprocess, "Popen", return_value=process),
                    patch.object(
                        trainer,
                        "publish_openwakeword_artifacts",
                        side_effect=RuntimeError(
                            "openWakeWord failed deployment calibration; "
                            "the existing published model was preserved"
                        ),
                    ),
                ):
                    rc = trainer._run_openwakeword_training(
                        phrase="hey tater",
                        safe_word="hey_tater",
                        log_path=log_path,
                    )

                self.assertEqual(rc, 1)
                saved_log = log_path.read_text(encoding="utf-8")
                self.assertIn("failed deployment calibration", saved_log)
                self.assertIn("Other selected model stages will continue", saved_log)
        finally:
            trainer.TRAINING_PROCESS = original_process


if __name__ == "__main__":
    unittest.main()
