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

    def test_active_training_process_group_stops_during_shutdown(self):
        original_root = trainer.ROOT_DIR
        original_data = trainer.DATA_DIR
        original_script = trainer.TRAIN_SCRIPT
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
                    "sleep 30\n",
                    encoding="utf-8",
                )
                trainer.ROOT_DIR = source
                trainer.DATA_DIR = data
                trainer.TRAIN_SCRIPT = str(script)

                thread = trainer._start_training_thread("hey_tater", "en", False, None)
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

                self.assertTrue(trainer._stop_training_runtime(timeout=3.0))
                thread.join(timeout=1.0)
                self.assertFalse(thread.is_alive())
                self.assertIsNotNone(proc.poll())
        finally:
            trainer.ROOT_DIR = original_root
            trainer.DATA_DIR = original_data
            trainer.TRAIN_SCRIPT = original_script

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
