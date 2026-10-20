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

            model_path, bundle_path, esphome_path = package_model.package_model(
                "Hey Tater",
                "en",
                calibration_path,
                root / "output",
                name_by_wake_word=True,
                source_model=source_model,
            )

            self.assertEqual(model_path.name, "hey_tater.tflite")
            self.assertEqual(bundle_path.name, "hey_tater.wake-bundle.json")
            self.assertEqual(esphome_path.name, "hey_tater.esphome.json")
            self.assertEqual(model_path.read_bytes(), b"model")

            tater_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
            micro_payload = tater_payload["micro_wake_word"]
            esphome_payload = json.loads(esphome_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(esphome_payload),
                set(package_model.ESPHOME_MANIFEST_KEYS),
            )
            self.assertEqual(esphome_payload["model"], "hey_tater.tflite")
            self.assertEqual(tater_payload["schema_version"], 3)
            self.assertEqual(esphome_payload["micro"], micro_payload["micro"])
            self.assertIn("tater_native", micro_payload)
            self.assertIn("calibration", micro_payload)
            self.assertFalse((bundle_path.parent / "hey_tater.json").exists())
            self.assertNotIn("tater_native", esphome_payload)
            self.assertNotIn("calibration", esphome_payload)

    def test_non_ascii_wake_word_keeps_phrase_and_uses_unique_artifact_slug(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_model = root / "source.tflite"
            source_model.write_bytes(b"model")

            model_path, bundle_path, esphome_path = package_model.package_model(
                "こんにちは タター",
                "ja",
                root / "missing-calibration.json",
                root / "output",
                name_by_wake_word=True,
                source_model=source_model,
            )

            expected_slug = package_model.safe_slug("こんにちは タター")
            self.assertRegex(expected_slug, r"^wakeword_[0-9a-f]{8}$")
            self.assertEqual(model_path.name, f"{expected_slug}.tflite")
            self.assertEqual(bundle_path.name, f"{expected_slug}.wake-bundle.json")
            self.assertEqual(esphome_path.name, f"{expected_slug}.esphome.json")
            payload = json.loads(bundle_path.read_text(encoding="utf-8"))["micro_wake_word"]
            self.assertEqual(payload["wake_word"], "こんにちは タター")
            self.assertEqual(payload["trained_languages"], ["ja"])


if __name__ == "__main__":
    unittest.main()
