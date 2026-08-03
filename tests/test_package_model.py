import json
import tempfile
import unittest
from pathlib import Path

from scripts_macos import package_model


class PackageModelTests(unittest.TestCase):
    def test_wake_word_package_contains_tater_and_esphome_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_model = root / "source.tflite"
            source_model.write_bytes(b"model")
            calibration_path = root / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "probability_cutoff": 0.98,
                        "sliding_window_size": 5,
                        "selected_metrics": {"recall": 0.99},
                    }
                ),
                encoding="utf-8",
            )

            model_path, json_path, esphome_path = package_model.package_model(
                "Hey Tater",
                "en",
                calibration_path,
                root / "output",
                name_by_wake_word=True,
                source_model=source_model,
            )

            self.assertEqual(model_path.name, "hey_tater.tflite")
            self.assertEqual(json_path.name, "hey_tater.json")
            self.assertEqual(esphome_path.name, "hey_tater.esphome.json")
            self.assertEqual(model_path.read_bytes(), b"model")

            tater_payload = json.loads(json_path.read_text(encoding="utf-8"))
            esphome_payload = json.loads(esphome_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(esphome_payload),
                set(package_model.ESPHOME_MANIFEST_KEYS),
            )
            self.assertEqual(esphome_payload["model"], "hey_tater.tflite")
            self.assertEqual(esphome_payload["micro"], tater_payload["micro"])
            self.assertIn("tater_native", tater_payload)
            self.assertIn("calibration", tater_payload)
            self.assertNotIn("tater_native", esphome_payload)
            self.assertNotIn("calibration", esphome_payload)


if __name__ == "__main__":
    unittest.main()
