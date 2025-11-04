# scripts_macos/make_features.py

import os

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

# Process TTS generated samples (default)
clips_tts = Clips(
    input_directory="./generated_samples",
    file_pattern="*.wav",
    max_clip_duration_s=5,
    remove_silence=True,
    random_split_seed=10,
    split_count=0.1,
)

# Process personal recordings if available (optional)
clips_personal = None
if os.path.exists("./personal_samples") and any(Path("./personal_samples").glob("*.wav")):
    clips_personal = Clips(
        input_directory="./personal_samples",
        file_pattern="*.wav",
        max_clip_duration_s=5,
        remove_silence=True,
        random_split_seed=10,
        split_count=0.1,
    )
    print("✅ Found personal samples, will create separate feature set")

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

# Process TTS samples
for split, cfg in split_cfg.items():
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"🧪 Processing {split} (TTS) …")
    spectros = SpectrogramGeneration(
        clips=clips_tts,
        augmenter=augmenter,
        slide_frames=cfg["slide_frames"],
        step_ms=10,
    )
    RaggedMmap.from_generator(
        out_dir=str(out_dir / "wakeword_mmap"),
<<<<<<< HEAD
        sample_generator=spectros.spectrogram_generator(
            split=cfg["name"],
            repeat=cfg["repetition"],
        ),
=======
        sample_generator=spectros.spectrogram_generator(split=cfg["name"], repeat=cfg["repetition"]),
>>>>>>> 426f158 (Undo)
        batch_size=100,
        verbose=True,
    )

# Process personal samples if available
if clips_personal is not None:
    out_root_personal = Path("personal_augmented_features")
    out_root_personal.mkdir(exist_ok=True)
    for split, cfg in split_cfg.items():
        out_dir = out_root_personal / split
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"🧪 Processing {split} (personal) …")
        spectros = SpectrogramGeneration(
            clips=clips_personal,
            augmenter=augmenter,
            slide_frames=cfg["slide_frames"],
            step_ms=10,
        )
        RaggedMmap.from_generator(
            out_dir=str(out_dir / "wakeword_mmap"),
            sample_generator=spectros.spectrogram_generator(split=cfg["name"], repeat=cfg["repetition"]),
            batch_size=100,
            verbose=True,
        )
print("✅ Features ready.")