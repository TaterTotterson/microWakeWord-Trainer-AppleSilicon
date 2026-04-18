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
sliding_window_size = 5
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

meta = {
  "type": "micro",
  "wake_word": wake,
  "author": "master phooey",
  "website": "https://github.com/MasterPhooey/MicroWakeWord-Trainer-Docker",
  "model": "stream_state_internal_quant.tflite",
  "trained_languages": [language],
  "version": 2,
  "micro": {
    "probability_cutoff": round(probability_cutoff, 2),
    "sliding_window_size": sliding_window_size,
    "feature_step_size": 10,
    "tensor_arena_size": 30000,
    "minimum_esphome_version": "2024.7.0"
  }
}
Path("stream_state_internal_quant.json").write_text(json.dumps(meta, indent=2))
print("📦 Wrote stream_state_internal_quant.tflite and stream_state_internal_quant.json")
