# scripts_macos/make_features.py

import os

# Force Hugging Face datasets to use soundfile and NOT torchcodec
# (this must run before importing anything that uses datasets.audio)
os.environ.setdefault("DATASETS_AUDIO_BACKEND", "soundfile")

from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration
from pathlib import Path


def validate(paths):
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"❌ Missing directory: {p}. Run dataset prep first.")


impulse_paths = ["mit_rirs"]
background_paths = ["fma_16k", "audioset_16k"]
validate(impulse_paths + background_paths)

clips = Clips(
    input_directory="./generated_samples",
    file_pattern="*.wav",
    max_clip_duration_s=5,
    remove_silence=True,
    random_split_seed=10,
    split_count=0.1,
)

augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={
        "SevenBandParametricEQ": 0.1,
        "TanhDistortion": 0.05,
        "PitchShift": 0.15,
        "BandStopFilter": 0.1,
        "AddColorNoise": 0.1,
        "AddBackgroundNoise": 0.7,
        "Gain": 0.8,
        "RIR": 0.7,
    },
    impulse_paths=impulse_paths,
    background_paths=background_paths,
    background_min_snr_db=5,
    background_max_snr_db=10,
    min_jitter_s=0.2,
    max_jitter_s=0.3,
)

out_root = Path("generated_augmented_features")
out_root.mkdir(exist_ok=True)

split_cfg = {
    "training":   {"name": "train",      "repetition": 2, "slide_frames": 10},
    "validation": {"name": "validation", "repetition": 1, "slide_frames": 10},
    "testing":    {"name": "test",       "repetition": 1, "slide_frames": 1},
}

for split, cfg in split_cfg.items():
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"🧪 Processing {split} …")
    spectros = SpectrogramGeneration(
        clips=clips,
        augmenter=augmenter,
        slide_frames=cfg["slide_frames"],
        step_ms=10,
    )
    RaggedMmap.from_generator(
        out_dir=str(out_dir / "wakeword_mmap"),
        sample_generator=spectros.spectrogram_generator(
            split=cfg["name"],
            repeat=cfg["repetition"],
        ),
        batch_size=100,
        verbose=True,
    )

print("✅ Features ready.")