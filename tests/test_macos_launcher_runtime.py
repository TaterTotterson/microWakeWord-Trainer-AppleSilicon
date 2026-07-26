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


if __name__ == "__main__":
    unittest.main()
