import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import call, patch

import trainer_server as trainer


class _FakeTrainingProcess:
    def __init__(self, *, timeout_once: bool = False):
        self.pid = 4321
        self.returncode = None
        self.timeout_once = timeout_once
        self.wait_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.timeout_once and self.wait_calls == 1:
            raise subprocess.TimeoutExpired(cmd="training", timeout=timeout)
        self.returncode = -signal.SIGKILL if self.timeout_once else 0
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1


class ShutdownTests(unittest.TestCase):
    def tearDown(self):
        trainer.TRAINING_SHUTDOWN_EVENT.clear()
        trainer.TRAINING_STOP_EVENT.clear()

    def test_training_process_group_gets_term_then_kill_after_timeout(self):
        proc = _FakeTrainingProcess(timeout_once=True)
        with (
            patch.object(trainer.os, "getpgid", return_value=proc.pid),
            patch.object(trainer.os, "getpgrp", return_value=999),
            patch.object(trainer.os, "killpg") as killpg,
        ):
            stopped = trainer._terminate_training_process_tree(
                proc,
                graceful_timeout=0.1,
                kill_timeout=0.1,
            )

        self.assertTrue(stopped)
        self.assertEqual(
            killpg.call_args_list,
            [
                call(proc.pid, signal.SIGTERM),
                call(proc.pid, signal.SIGKILL),
            ],
        )
        self.assertEqual(proc.terminate_calls, 0)
        self.assertEqual(proc.kill_calls, 0)

    def test_new_training_is_rejected_after_shutdown_starts(self):
        trainer.TRAINING_SHUTDOWN_EVENT.set()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            trainer._start_training_thread("hey_tater", "en", False, None)

    def test_session_stop_terminates_training_without_shutting_down_server(self):
        proc = _FakeTrainingProcess()
        original_process = trainer.TRAINING_PROCESS
        original_thread = trainer.TRAINING_THREAD
        try:
            trainer.TRAINING_PROCESS = proc
            trainer.TRAINING_THREAD = None
            with (
                patch.object(trainer.os, "getpgid", return_value=proc.pid),
                patch.object(trainer.os, "getpgrp", return_value=999),
                patch.object(trainer.os, "killpg") as killpg,
            ):
                self.assertTrue(trainer._stop_current_training(timeout=0.2))
            killpg.assert_called_once_with(proc.pid, signal.SIGTERM)
            self.assertFalse(trainer.TRAINING_SHUTDOWN_EVENT.is_set())
            self.assertFalse(trainer.TRAINING_STOP_EVENT.is_set())
        finally:
            trainer.TRAINING_PROCESS = original_process
            trainer.TRAINING_THREAD = original_thread

    def test_active_training_process_group_stops_during_shutdown(self):
        original_root = trainer.ROOT_DIR
        original_data = trainer.DATA_DIR
        original_script = trainer.TRAIN_SCRIPT
        original_raw_phrase = trainer.STATE.get("raw_phrase")
        trainer.TRAINING_SHUTDOWN_EVENT.clear()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                source = root / "source"
                data = root / "data"
                source.mkdir()
                data.mkdir()
                script = source / "train.sh"
                script.write_text(
                    "#!/bin/sh\n"
                    "printf '%s\\n' \"$PWD\" > \"$WAKEWORD_TRAINER_DATA_DIR/working-directory.txt\"\n"
                    "printf '%s\\n%s\\n%s\\n' \"$1\" \"$MWW_ARTIFACT_SLUG\" \"$MWW_LANGUAGE\" "
                    "> \"$WAKEWORD_TRAINER_DATA_DIR/training-identity.txt\"\n"
                    "sleep 30\n",
                    encoding="utf-8",
                )
                trainer.ROOT_DIR = source
                trainer.DATA_DIR = data
                trainer.TRAIN_SCRIPT = str(script)
                raw_phrase = "こんにちは タター"
                safe_word = trainer.safe_name(raw_phrase)
                with trainer.STATE_LOCK:
                    trainer.STATE["raw_phrase"] = raw_phrase

                thread = trainer._start_training_thread(
                    safe_word,
                    "ja",
                    False,
                    None,
                    tts_mode="modern",
                )
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    with trainer.TRAINING_RUNTIME_LOCK:
                        proc = trainer.TRAINING_PROCESS
                    if proc is not None:
                        break
                    time.sleep(0.02)
                else:
                    self.fail("Training subprocess did not start.")

                working_directory_file = data / "working-directory.txt"
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and not working_directory_file.exists():
                    time.sleep(0.02)
                self.assertEqual(
                    Path(working_directory_file.read_text(encoding="utf-8").strip()).resolve(),
                    data.resolve(),
                )
                self.assertTrue((data / "recorder_training.log").exists())
                self.assertFalse((source / "recorder_training.log").exists())
                identity_file = data / "training-identity.txt"
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and not identity_file.exists():
                    time.sleep(0.02)
                self.assertEqual(
                    identity_file.read_text(encoding="utf-8").splitlines(),
                    [raw_phrase, safe_word, "ja"],
                )

                self.assertTrue(trainer._stop_training_runtime(timeout=3.0))
                thread.join(timeout=1.0)
                self.assertFalse(thread.is_alive())
                self.assertIsNotNone(proc.poll())
        finally:
            trainer.ROOT_DIR = original_root
            trainer.DATA_DIR = original_data
            trainer.TRAIN_SCRIPT = original_script
            with trainer.STATE_LOCK:
                trainer.STATE["raw_phrase"] = original_raw_phrase

    def test_non_ascii_phrase_gets_deterministic_unique_slug(self):
        slug = trainer.safe_name("こんにちは タター")
        self.assertRegex(slug, r"^wakeword_[0-9a-f]{8}$")
        self.assertEqual(slug, trainer.safe_name("こんにちは タター"))
        self.assertNotEqual(slug, trainer.safe_name("おはよう タター"))

    def test_server_shutdown_stops_scheduler_before_training_runtime(self):
        with (
            patch.object(trainer, "_stop_auto_train_worker", return_value=True) as stop_worker,
            patch.object(trainer, "_stop_training_runtime", return_value=True) as stop_training,
        ):
            trainer.stop_auto_train_worker_event()

        stop_worker.assert_called_once_with(timeout=5.0)
        stop_training.assert_called_once_with(timeout=20.0)


if __name__ == "__main__":
    unittest.main()
