import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_SOURCE = (
    ROOT
    / "macos"
    / "WakeWordTrainer"
    / "Sources"
    / "WakeWordTrainer"
    / "main.swift"
)
RUN_SCRIPT = ROOT / "run.sh"
TRAINER_SOURCE = ROOT / "trainer_server.py"


class MacOSLauncherRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = LAUNCHER_SOURCE.read_text(encoding="utf-8")
        cls.run_script = RUN_SCRIPT.read_text(encoding="utf-8")

    def test_output_handlers_detach_at_eof(self):
        eof_guards = re.findall(
            r"guard !data\.isEmpty else \{\s*"
            r"reader\.readabilityHandler = nil\s*"
            r"return\s*"
            r"\}",
            self.launcher,
            re.MULTILINE,
        )
        self.assertGreaterEqual(len(eof_guards), 2)

    def test_managed_process_detaches_pipe_before_final_wait(self):
        start = self.launcher.index("private func waitForManagedProcessExit")
        end = self.launcher.index("private func terminateBackendProcessNow", start)
        implementation = self.launcher[start:end]

        self.assertLess(
            implementation.index("detachOutputPipe()"),
            implementation.index("process.waitUntilExit()"),
        )

    def test_detach_helper_clears_handler_and_reference(self):
        start = self.launcher.index("private func detachOutputPipe")
        end = self.launcher.index("private func appendLog", start)
        implementation = self.launcher[start:end]

        self.assertIn(
            "outputPipe?.fileHandleForReading.readabilityHandler = nil",
            implementation,
        )
        self.assertIn("outputPipe = nil", implementation)

    def test_dependency_list_drives_install_and_fingerprint(self):
        self.assertIn("UI_DEPENDENCIES=(", self.run_script)
        self.assertIn('$PIP install "${UI_DEPENDENCIES[@]}"', self.run_script)
        self.assertIn(
            "printf '%s\\n' \"${UI_DEPENDENCIES[@]}\" | /usr/bin/shasum -a 256",
            self.run_script,
        )
        self.assertIn(
            'INSTALLED_DEPENDENCY_FINGERPRINT" != "$EXPECTED_DEPENDENCY_FINGERPRINT',
            self.run_script,
        )
        self.assertIn('dependency_fingerprint > "$PIN_FILE"', self.run_script)
        self.assertNotIn('touch "$PIN_FILE"', self.run_script)

    def test_trainer_source_compiles(self):
        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        compile(source, str(TRAINER_SOURCE), "exec")

    def test_app_updates_preserve_modern_tts_caches(self):
        self.assertIn('"tts-envs/"', self.launcher)
        self.assertIn('"voice-bank/"', self.launcher)

    def test_termination_reply_runs_in_appkit_modal_run_loop(self):
        start = self.launcher.index("func applicationShouldTerminate(")
        end = self.launcher.index("private func startRecoveryWatchdog", start)
        implementation = self.launcher[start:end]

        self.assertIn("CFRunLoopPerformBlock(", implementation)
        self.assertIn(
            "RunLoop.Mode.modalPanel.rawValue as CFString",
            implementation,
        )
        self.assertIn("CFRunLoopWakeUp(mainRunLoop)", implementation)
        self.assertNotIn("DispatchQueue.main.async {", implementation)

    def test_installer_forces_stuck_old_app_to_exit_before_replacing_it(self):
        start = self.launcher.index("private func writeInstallerScript()")
        end = self.launcher.index("private func safePathComponent", start)
        implementation = self.launcher[start:end]

        term_index = implementation.index('kill -TERM "$APP_PID"')
        kill_index = implementation.index('kill -KILL "$APP_PID"')
        target_index = implementation.index(
            'TARGET_PARENT="$(dirname "$TARGET_APP")"'
        )
        self.assertLess(term_index, kill_index)
        self.assertLess(kill_index, target_index)


if __name__ == "__main__":
    unittest.main()
