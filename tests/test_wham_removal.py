import unittest
from pathlib import Path


class WhamRemovalTests(unittest.TestCase):
    def test_active_training_pipeline_has_no_wham_dependency(self):
        root = Path(__file__).resolve().parents[1]
        active_files = [
            root / "scripts_macos" / "prepare_datasets.py",
            root / "scripts_macos" / "make_features.py",
            root / "train_microwakeword_macos.sh",
        ]

        for path in active_files:
            with self.subTest(path=path.name):
                self.assertNotIn("wham", path.read_text().lower())


if __name__ == "__main__":
    unittest.main()
