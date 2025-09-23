# scripts_macos/package_model.py
import shutil, json, sys, os
from pathlib import Path

wake = sys.argv[1] if len(sys.argv) > 1 else "hey_norman"
src = Path("trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite")
dst = Path("stream_state_internal_quant.tflite")
if not src.exists():
    raise SystemExit(f"❌ Model not found at {src}")

shutil.copy(src, dst)

meta = {
  "type": "micro",
  "wake_word": wake,
  "author": "master phooey",
  "website": "https://github.com/MasterPhooey/MicroWakeWord-Trainer-Docker",
  "model": "stream_state_internal_quant.tflite",
  "trained_languages": ["en"],
  "version": 2,
  "micro": {
    "probability_cutoff": 0.97,
    "sliding_window_size": 5,
    "feature_step_size": 10,
    "tensor_arena_size": 30000,
    "minimum_esphome_version": "2024.7.0"
  }
}
Path("stream_state_internal_quant.json").write_text(json.dumps(meta, indent=2))
print("📦 Wrote stream_state_internal_quant.tflite and stream_state_internal_quant.json")