#!/usr/bin/env python3
"""Check packaged wakeword model against raw personal and negative WAV files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import wavfile

from microwakeword.inference import Model


def _read_wav_16k_mono(path: Path) -> np.ndarray:
    sample_rate, data = wavfile.read(path)
    if sample_rate != 16000:
        raise ValueError(f"{path}: sample rate must be 16000Hz (got {sample_rate})")
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.floating):
        data = np.clip(data, -1.0, 1.0)
        data = (data * 32767).astype(np.int16)
    elif data.dtype != np.int16:
        data = data.astype(np.int16)
    return data


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    if values.size < window:
        return values
    return sliding_window_view(values, window).mean(axis=1)


def _score(
    model_path: Path,
    wav_path: Path,
    window: int,
    step_ms: int,
    stride: int | None,
) -> tuple[float, float, int]:
    model = Model(str(model_path), stride=stride)
    pcm = _read_wav_16k_mono(wav_path)
    probabilities = np.asarray(model.predict_clip(pcm, step_ms=step_ms), dtype=np.float32)
    if probabilities.size == 0:
        return 0.0, 0.0, 0
    smoothed = _moving_average(probabilities, window)
    max_prob = float(np.max(probabilities))
    max_smoothed = float(np.max(smoothed)) if smoothed.size else max_prob
    return max_prob, max_smoothed, int(probabilities.size)


def _iter_wavs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.glob("*.wav"))


def _infer_stride(training_config: Path, explicit_stride: int | None) -> tuple[int | None, str]:
    if explicit_stride is not None:
        return explicit_stride, "cli"
    if training_config.exists():
        try:
            config = yaml.load(training_config.read_text(encoding="utf-8"), Loader=yaml.Loader)
            if "stride" in config:
                return int(config["stride"]), str(training_config)
            flags = config.get("flags", {})
            if "stride" in flags:
                return int(flags["stride"]), str(training_config)
        except Exception as exc:
            print(f"⚠️ Could not infer stride from {training_config}: {exc}")
    return None, "model_input_slices"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check raw wakeword samples")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--positive-dir", type=Path, default=Path("personal_samples"))
    parser.add_argument("--negative-dir", type=Path, default=Path("negative_samples"))
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("trained_models/wakeword/training_config.yaml"),
    )
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--step-ms", type=int, default=10)
    parser.add_argument(
        "--fail-on-negative-detect",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-on-low-positive-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--min-positive-detection-rate",
        type=float,
        default=float(os.environ.get("MWW_RAW_POSITIVE_MIN_DETECTION_RATE", "0.80")),
        help="Minimum required positive detection rate before the raw sample check fails.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.min_positive_detection_rate <= 1.0:
        raise ValueError("--min-positive-detection-rate must be between 0.0 and 1.0")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    micro = manifest.get("micro", {})
    cutoff = float(micro.get("probability_cutoff", 0.97))
    window = int(micro.get("sliding_window_size", 5))
    model_path = args.manifest.parent / str(manifest["model"])
    stride, stride_source = _infer_stride(args.training_config, args.stride)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found from manifest: {model_path}")

    print(
        f"===== Raw Sample Check =====\n"
        f"model={model_path} cutoff={cutoff:.2f} window={window} "
        f"step_ms={args.step_ms} stride={stride if stride is not None else 'auto'} "
        f"({stride_source}) rule=ma_max>cutoff"
    )

    negative_failures = 0
    positive_failures = 0
    for label, directory in (
        ("positive", args.positive_dir),
        ("negative", args.negative_dir),
    ):
        wavs = _iter_wavs(directory)
        if not wavs:
            print(f"ℹ️ No {label} wavs found in {directory}; skipping.")
            continue

        detected_count = 0
        for wav_path in wavs:
            max_prob, max_smoothed, frames = _score(
                model_path=model_path,
                wav_path=wav_path,
                window=window,
                step_ms=args.step_ms,
                stride=stride,
            )
            detected = max_smoothed > cutoff
            detected_count += int(detected)
            status = "DETECT" if detected else "MISS"
            print(
                f"{label:8s} {status:6s} max={max_prob:.4f} "
                f"ma_max={max_smoothed:.4f} frames={frames:5d} file={wav_path}"
            )
            if label == "negative" and detected and args.fail_on_negative_detect:
                negative_failures += 1

        print(f"{label} summary: {detected_count}/{len(wavs)} detected")
        if label == "positive" and args.fail_on_low_positive_detection:
            detection_rate = detected_count / len(wavs)
            if detection_rate < args.min_positive_detection_rate:
                positive_failures = len(wavs) - detected_count
                print(
                    "❌ Raw positive check failed: "
                    f"{detected_count}/{len(wavs)} detected "
                    f"({detection_rate:.1%}), required "
                    f"{args.min_positive_detection_rate:.1%}."
                )

    if negative_failures:
        print(
            f"❌ Raw negative check failed: {negative_failures} negative sample(s) "
            "still trigger the wakeword."
        )
        return 1
    if positive_failures:
        return 1

    print("✅ Raw sample check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
