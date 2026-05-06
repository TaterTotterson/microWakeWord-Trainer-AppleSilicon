#!/usr/bin/env python3
"""Tune detector metadata against raw personal and negative WAV samples."""

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


def _score_files(
    model_path: Path,
    wavs: list[Path],
    windows: list[int],
    step_ms: int,
    stride: int | None,
) -> dict[int, list[tuple[Path, float]]]:
    scores = {window: [] for window in windows}
    for index, wav_path in enumerate(wavs, start=1):
        model = Model(str(model_path), stride=stride)
        probabilities = np.asarray(
            model.predict_clip(_read_wav_16k_mono(wav_path), step_ms=step_ms),
            dtype=np.float32,
        )
        for window in windows:
            smoothed = _moving_average(probabilities, window)
            max_score = float(np.max(smoothed)) if smoothed.size else 0.0
            scores[window].append((wav_path, max_score))
        if index % 10 == 0 or index == len(wavs):
            print(f"   scored {index}/{len(wavs)} raw file(s)")
    return scores


def _candidate_cutoffs() -> list[float]:
    return [step / 100 for step in range(100, -1, -1)]


def _is_detected(score: float, cutoff: float) -> bool:
    # Match microWakeWord detector calibration semantics: probability > cutoff.
    return score > cutoff


def _count_detected(values: list[tuple[Path, float]], cutoff: float) -> int:
    return sum(_is_detected(score, cutoff) for _, score in values)


def _format_ranked_scores(values: list[tuple[Path, float]], *, reverse: bool, limit: int) -> list[str]:
    ranked = sorted(values, key=lambda item: item[1], reverse=reverse)[:limit]
    return [f"{path.name}={score:.4f}" for path, score in ranked]


def _ranked_score_records(
    values: list[tuple[Path, float]], *, reverse: bool, limit: int
) -> list[dict[str, object]]:
    ranked = sorted(values, key=lambda item: item[1], reverse=reverse)[:limit]
    return [{"file": str(path), "score": score} for path, score in ranked]


def _build_tradeoffs(
    positive_scores: dict[int, list[tuple[Path, float]]],
    negative_scores: dict[int, list[tuple[Path, float]]],
    windows: list[int],
) -> list[dict[str, object]]:
    tradeoffs = []
    for window in windows:
        positive_values = positive_scores[window]
        negative_values = negative_scores[window]
        for cutoff in _candidate_cutoffs():
            positive_detected = _count_detected(positive_values, cutoff)
            negative_detected = _count_detected(negative_values, cutoff)
            tradeoffs.append(
                {
                    "window": window,
                    "probability_cutoff": cutoff,
                    "positive_detected": positive_detected,
                    "negative_detected": negative_detected,
                }
            )
    return tradeoffs


def _best_required_positive_tradeoffs(
    tradeoffs: list[dict[str, object]],
    required_positive_count: int,
    *,
    limit: int = 10,
) -> list[dict[str, object]]:
    viable_positive = [
        item for item in tradeoffs if item["positive_detected"] >= required_positive_count
    ]
    return sorted(
        viable_positive,
        key=lambda item: (
            item["negative_detected"],
            -item["positive_detected"],
            -item["probability_cutoff"],
            item["window"],
        ),
    )[:limit]


def _best_zero_negative_attempts(
    tradeoffs: list[dict[str, object]], *, limit: int = 10
) -> list[dict[str, object]]:
    zero_negative = [item for item in tradeoffs if item["negative_detected"] == 0]
    return sorted(
        zero_negative,
        key=lambda item: (
            -item["positive_detected"],
            -item["probability_cutoff"],
            item["window"],
        ),
    )[:limit]


def _write_failure_diagnostics(
    output_path: Path,
    *,
    model_path: Path,
    positive_scores: dict[int, list[tuple[Path, float]]],
    negative_scores: dict[int, list[tuple[Path, float]]],
    windows: list[int],
    required_positive_count: int,
    top_k: int,
    tradeoffs: list[dict[str, object]],
) -> None:
    diagnostics = {
        "model": str(model_path),
        "detector_rule": "score > probability_cutoff",
        "required_positive_count": required_positive_count,
        "positive_total": len(next(iter(positive_scores.values()), [])),
        "negative_total": len(next(iter(negative_scores.values()), [])),
        "windows": {},
        "best_required_positive_tradeoffs": _best_required_positive_tradeoffs(
            tradeoffs,
            required_positive_count,
        ),
        "best_zero_negative_attempts": _best_zero_negative_attempts(tradeoffs),
    }
    for window in windows:
        positive_values = positive_scores[window]
        negative_values = negative_scores[window]
        max_negative = max((score for _, score in negative_values), default=0.0)
        diagnostics["windows"][str(window)] = {
            "max_negative_score": max_negative,
            "positives_above_max_negative": _count_detected(positive_values, max_negative),
            "top_negative_scores": _ranked_score_records(
                negative_values,
                reverse=True,
                limit=top_k,
            ),
            "weakest_positive_scores": _ranked_score_records(
                positive_values,
                reverse=False,
                limit=top_k,
            ),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


def _print_failure_diagnostics(
    *,
    positive_scores: dict[int, list[tuple[Path, float]]],
    negative_scores: dict[int, list[tuple[Path, float]]],
    windows: list[int],
    required_positive_count: int,
    top_k: int,
    tradeoffs: list[dict[str, object]],
    diagnostics_output: Path,
) -> None:
    positive_total = len(next(iter(positive_scores.values()), []))
    for window in windows:
        positive_values = positive_scores[window]
        negative_values = negative_scores[window]
        max_negative = max((score for _, score in negative_values), default=0.0)
        positives_above = _count_detected(positive_values, max_negative)
        print(
            f"   window={window}: max_negative={max_negative:.4f}, "
            f"positives_above_max_negative={positives_above}/{positive_total}"
        )
        print(
            "      blockers: "
            + ", ".join(_format_ranked_scores(negative_values, reverse=True, limit=top_k))
        )
        print(
            "      weakest positives: "
            + ", ".join(_format_ranked_scores(positive_values, reverse=False, limit=top_k))
        )

    zero_negative_attempts = _best_zero_negative_attempts(tradeoffs, limit=3)
    if zero_negative_attempts:
        print("   Best zero-negative attempts:")
        for item in zero_negative_attempts:
            print(
                f"      window={item['window']} cutoff={item['probability_cutoff']:.2f} "
                f"positive={item['positive_detected']} negative={item['negative_detected']}"
            )

    best_tradeoffs = _best_required_positive_tradeoffs(
        tradeoffs,
        required_positive_count,
    )
    if best_tradeoffs:
        print(f"   Best tradeoffs with >= {required_positive_count} positives:")
        for item in best_tradeoffs:
            print(
                f"      window={item['window']} cutoff={item['probability_cutoff']:.2f} "
                f"positive={item['positive_detected']} negative={item['negative_detected']}"
            )
    print(f"📝 Wrote raw tuning diagnostics: {diagnostics_output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune raw detector settings")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("trained_models/wakeword/training_config.yaml"),
    )
    parser.add_argument("--positive-dir", type=Path, default=Path("personal_samples"))
    parser.add_argument("--negative-dir", type=Path, default=Path("negative_samples"))
    parser.add_argument("--windows", type=str, default="3,4,5,6,7")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--step-ms", type=int, default=10)
    parser.add_argument("--diagnostics-output", type=Path, default=None)
    parser.add_argument("--top-k-diagnostics", type=int, default=5)
    parser.add_argument(
        "--min-positive-detection-rate",
        type=float,
        default=float(os.environ.get("MWW_RAW_POSITIVE_MIN_DETECTION_RATE", "0.80")),
    )
    args = parser.parse_args()

    if not 0.0 <= args.min_positive_detection_rate <= 1.0:
        raise ValueError("--min-positive-detection-rate must be between 0.0 and 1.0")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    windows = [int(value) for value in args.windows.split(",") if value.strip()]
    positives = _iter_wavs(args.positive_dir)
    negatives = _iter_wavs(args.negative_dir)
    if not positives or not negatives:
        print("ℹ️ Raw tuning skipped; both positive and negative WAVs are required.")
        return 0

    stride, stride_source = _infer_stride(args.training_config, args.stride)
    print(
        "===== Raw Detector Tuning =====\n"
        f"model={args.model} positives={len(positives)} negatives={len(negatives)} "
        f"windows={windows} stride={stride if stride is not None else 'auto'} "
        f"({stride_source})"
    )

    positive_scores = _score_files(args.model, positives, windows, args.step_ms, stride)
    negative_scores = _score_files(args.model, negatives, windows, args.step_ms, stride)

    required_positive_count = int(np.ceil(len(positives) * args.min_positive_detection_rate))
    tradeoffs = _build_tradeoffs(positive_scores, negative_scores, windows)
    candidates = []
    for item in tradeoffs:
        if (
            item["negative_detected"] == 0
            and item["positive_detected"] >= required_positive_count
        ):
            candidates.append(
                (
                    item["positive_detected"],
                    item["probability_cutoff"],
                    item["window"],
                )
            )

    if not candidates:
        diagnostics_output = args.diagnostics_output or args.calibration.with_name(
            "detection_raw_tuning_failure.json"
        )
        _write_failure_diagnostics(
            diagnostics_output,
            model_path=args.model,
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            windows=windows,
            required_positive_count=required_positive_count,
            top_k=args.top_k_diagnostics,
            tradeoffs=tradeoffs,
        )
        print(
            "❌ No raw detector setting satisfies "
            f"{required_positive_count}/{len(positives)} positives with 0/{len(negatives)} negatives."
        )
        _print_failure_diagnostics(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
            windows=windows,
            required_positive_count=required_positive_count,
            top_k=args.top_k_diagnostics,
            tradeoffs=tradeoffs,
            diagnostics_output=diagnostics_output,
        )
        return 1

    positive_detected, cutoff, window = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
    print(
        "✓ Raw detector tuning selected "
        f"cutoff={cutoff:.2f}, window={window}, "
        f"positive={positive_detected}/{len(positives)}, negative=0/{len(negatives)}"
    )

    calibration = {}
    if args.calibration.exists():
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration.update(
        {
            "probability_cutoff": cutoff,
            "sliding_window_size": window,
            "raw_positive_detected": positive_detected,
            "raw_positive_total": len(positives),
            "raw_negative_detected": 0,
            "raw_negative_total": len(negatives),
            "raw_min_positive_detection_rate": args.min_positive_detection_rate,
        }
    )
    args.calibration.parent.mkdir(parents=True, exist_ok=True)
    args.calibration.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(f"📝 Updated calibration with raw detector settings: {args.calibration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
