#!/usr/bin/env python3
"""Package a trained microWakeWord model for Tater Native and ESPHome."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path(
    "trained_models/wakeword/tflite_stream_state_internal_quant/"
    "stream_state_internal_quant.tflite"
)
DEFAULT_CALIBRATION_PATH = Path(
    "trained_models/wakeword/tflite_stream_state_internal_quant/"
    "detection_calibration.json"
)
ESPHOME_MANIFEST_KEYS = (
    "type",
    "wake_word",
    "author",
    "website",
    "model",
    "trained_languages",
    "version",
    "micro",
)


def safe_slug(wake_word: str) -> str:
    normalized = unicodedata.normalize("NFKC", wake_word or "").strip().lower()
    slug = re.sub(r"[^a-z0-9_]+", "", re.sub(r"\s+", "_", normalized))
    slug = re.sub(r"^_+|_+$", "", slug)
    if slug:
        return slug
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
    return f"wakeword_{digest}"


def esphome_manifest(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return only fields accepted by ESPHome's micro_wake_word v2 schema."""
    return {key: metadata[key] for key in ESPHOME_MANIFEST_KEYS if key in metadata}


def read_calibration(path: Path) -> tuple[dict[str, Any], float, int]:
    probability_cutoff = 0.97
    sliding_window_size = 6
    calibration: dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            calibration = payload if isinstance(payload, dict) else {}
            probability_cutoff = float(
                calibration.get("probability_cutoff", probability_cutoff)
            )
            sliding_window_size = int(
                calibration.get("sliding_window_size", sliding_window_size)
            )
            print(
                "🎯 Using calibrated detector settings: "
                f"cutoff={probability_cutoff:.2f}, window={sliding_window_size}"
            )
        except Exception as exc:
            print(f"⚠️ Failed to read detector calibration ({exc}); using defaults.")
            calibration = {}

    return (
        calibration,
        round(probability_cutoff, 3),
        max(1, min(10, int(sliding_window_size))),
    )


def package_model(
    wake_word: str,
    language: str,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    output_dir: Path = Path("."),
    *,
    name_by_wake_word: bool = False,
    artifact_slug: str = "",
    source_model: Path = DEFAULT_MODEL_PATH,
) -> tuple[Path, Path, Path]:
    if not source_model.exists():
        raise SystemExit(f"❌ Model not found at {source_model}")

    output_dir.mkdir(parents=True, exist_ok=True)
    basename = (
        safe_slug(artifact_slug or wake_word)
        if name_by_wake_word
        else "stream_state_internal_quant"
    )
    model_path = output_dir / f"{basename}.tflite"
    json_path = output_dir / f"{basename}.json"
    esphome_json_path = output_dir / f"{basename}.esphome.json"
    shutil.copy(source_model, model_path)

    calibration, probability_cutoff, sliding_window_size = read_calibration(
        calibration_path
    )
    selected_metrics = (
        calibration.get("selected_metrics")
        if isinstance(calibration.get("selected_metrics"), dict)
        else {}
    )
    evaluation = (
        calibration.get("evaluation")
        if isinstance(calibration.get("evaluation"), dict)
        else {}
    )
    close_miss_threshold = max(
        0.01,
        min(0.99, round(max(0.68, probability_cutoff - 0.17), 3)),
    )

    metadata = {
        "type": "micro",
        "wake_word": wake_word,
        "label": wake_word.replace("_", " ").title(),
        "author": "Tater Totterson",
        "website": (
            "https://github.com/TaterTotterson/"
            "microWakeWord-Trainer-AppleSilicon"
        ),
        "model": model_path.name,
        "trained_languages": [language],
        "version": 2,
        "model_format": "tflite_stream_state_internal_quant",
        "quantization": "int8",
        "sample_rate": 16000,
        "micro": {
            "probability_cutoff": probability_cutoff,
            "sliding_window_size": sliding_window_size,
            "feature_step_size": 10,
            "tensor_arena_size": 30000,
            "minimum_esphome_version": "2024.7.0",
        },
        "tater_native": {
            "format_version": 1,
            "wake_threshold": probability_cutoff,
            "wake_sliding_window": sliding_window_size,
            "close_miss_threshold": close_miss_threshold,
            "frontend": {
                "name": "tflm_microfrontend",
                "sample_rate": 16000,
                "feature_duration_ms": 30,
                "feature_step_ms": 10,
                "feature_size": 40,
                "input_feature_frames": 2,
                "lower_band_limit": 125.0,
                "upper_band_limit": 7500.0,
            },
            "recommended_for": ["tater-native-satellite", "voice-pe"],
        },
        "calibration": {
            "target_false_accepts_per_hour": calibration.get(
                "target_false_accepts_per_hour"
            ),
            "selected_false_accepts_per_hour_limit": calibration.get(
                "selected_false_accepts_per_hour_limit"
            ),
            "recall": selected_metrics.get("recall"),
            "false_accepts_per_hour": selected_metrics.get(
                "false_accepts_per_hour"
            ),
            "ambient_hours": selected_metrics.get("ambient_hours"),
            "positive_dataset": evaluation.get("positive_dataset"),
            "ambient_dataset": evaluation.get("ambient_dataset"),
            "positive_tracks": evaluation.get("positive_tracks"),
            "ambient_tracks": evaluation.get("ambient_tracks"),
            "generated_at": calibration.get("generated_at"),
        },
    }

    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    esphome_json_path.write_text(
        json.dumps(esphome_manifest(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"📦 Wrote {model_path}, {json_path}, and {esphome_json_path} "
        f"(wake word: {wake_word!r})"
    )
    return model_path, json_path, esphome_json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wake_word", nargs="?", default="hey_norman")
    parser.add_argument(
        "language",
        nargs="?",
        default=os.environ.get("MWW_LANGUAGE", "en"),
    )
    parser.add_argument(
        "calibration",
        nargs="?",
        default=os.environ.get("MWW_CALIBRATION_JSON", ""),
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--artifact-slug", default="")
    parser.add_argument("--name-by-wake-word", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_path = (
        Path(args.calibration) if args.calibration else DEFAULT_CALIBRATION_PATH
    )
    package_model(
        str(args.wake_word),
        str(args.language).strip().lower() or "en",
        calibration_path,
        Path(args.output_dir),
        name_by_wake_word=bool(args.name_by_wake_word),
        artifact_slug=str(args.artifact_slug),
    )


if __name__ == "__main__":
    main()
