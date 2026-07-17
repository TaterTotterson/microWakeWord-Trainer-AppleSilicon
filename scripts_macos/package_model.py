# scripts_macos/package_model.py
import shutil, json, sys, os
from pathlib import Path

wake = sys.argv[1] if len(sys.argv) > 1 else "hey_norman"
language = (sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MWW_LANGUAGE", "en")).strip().lower() or "en"
calibration_arg = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("MWW_CALIBRATION_JSON", "")
calibration_path = Path(calibration_arg) if calibration_arg else Path(
    "trained_models/wakeword/tflite_stream_state_internal_quant/detection_calibration.json"
)
src = Path("trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite")
dst = Path("stream_state_internal_quant.tflite")
if not src.exists():
    raise SystemExit(f"❌ Model not found at {src}")

shutil.copy(src, dst)

probability_cutoff = 0.97
sliding_window_size = 6
calibration = {}
if calibration_path.exists():
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        probability_cutoff = float(calibration.get("probability_cutoff", probability_cutoff))
        sliding_window_size = int(calibration.get("sliding_window_size", sliding_window_size))
        print(
            f"🎯 Using calibrated detector settings: "
            f"cutoff={probability_cutoff:.2f}, window={sliding_window_size}"
        )
    except Exception as exc:
        print(f"⚠️ Failed to read detector calibration ({exc}); using defaults.")

probability_cutoff = round(probability_cutoff, 3)
sliding_window_size = max(1, min(10, int(sliding_window_size)))
selected_metrics = calibration.get("selected_metrics") if isinstance(calibration.get("selected_metrics"), dict) else {}
evaluation = calibration.get("evaluation") if isinstance(calibration.get("evaluation"), dict) else {}
close_miss_threshold = max(
    0.01,
    min(0.99, round(max(0.68, probability_cutoff - 0.17), 3)),
)

meta = {
  "type": "micro",
  "wake_word": wake,
  "label": wake.replace("_", " ").title(),
  "author": "Tater Totterson",
  "website": "https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon",
  "model": "stream_state_internal_quant.tflite",
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
    "minimum_esphome_version": "2024.7.0"
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
      "upper_band_limit": 7500.0
    },
    "recommended_for": ["tater-native-satellite", "voice-pe"]
  },
  "calibration": {
    "target_false_accepts_per_hour": calibration.get("target_false_accepts_per_hour"),
    "selected_false_accepts_per_hour_limit": calibration.get("selected_false_accepts_per_hour_limit"),
    "recall": selected_metrics.get("recall"),
    "false_accepts_per_hour": selected_metrics.get("false_accepts_per_hour"),
    "ambient_hours": selected_metrics.get("ambient_hours"),
    "positive_dataset": evaluation.get("positive_dataset"),
    "ambient_dataset": evaluation.get("ambient_dataset"),
    "positive_tracks": evaluation.get("positive_tracks"),
    "ambient_tracks": evaluation.get("ambient_tracks"),
    "generated_at": calibration.get("generated_at")
  }
}
Path("stream_state_internal_quant.json").write_text(json.dumps(meta, indent=2))
print("📦 Wrote stream_state_internal_quant.tflite and stream_state_internal_quant.json")
