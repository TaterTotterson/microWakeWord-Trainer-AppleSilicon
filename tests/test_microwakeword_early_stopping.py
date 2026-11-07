from __future__ import annotations

import os
import runpy
import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


ROOT = Path(__file__).resolve().parents[1]


class MicroWakeWordEarlyStoppingConfigTests(unittest.TestCase):
    @unittest.skipIf(yaml is None, "PyYAML is installed by the trainer environment")
    def test_primary_config_enables_conservative_early_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                runpy.run_path(str(ROOT / "scripts_macos" / "write_training_yaml.py"))
                config = yaml.safe_load(
                    Path("training_parameters.yaml").read_text(encoding="utf-8")
                )
            finally:
                os.chdir(previous)

        self.assertEqual(config["training_steps"], [40_000])
        self.assertEqual(
            config["early_stopping"],
            {
                "enabled": True,
                "min_steps": 20_000,
                "patience_steps": 6_000,
                "min_delta": 0.001,
            },
        )


if __name__ == "__main__":
    unittest.main()
