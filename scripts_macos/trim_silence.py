#!/usr/bin/env python3
"""
Trim silence from the end of wave files using silero-vad.

For wave files in personal_samples (and any subdirectories), this script:
- Uses silero-vad to detect voice activity
- Truncates only from the end of the file
- Keeps only up to ~20ms of silence, with a 10ms jitter

Usage:
    python3 trim_silence.py
"""

import os
import random
import librosa
import soundfile as sf
from pathlib import Path
from silero_vad import load_silero_vad, get_speech_timestamps


def trim_silence(
    input_dir: str = 'personal_samples',
    keep_silence_duration: float = 0.020,
    jitter_enabled: bool = True
) -> None:
    """
    Trim silence from the end of all wave files in input_dir using silero-vad.

    Args:
        input_dir: Directory containing wave files (traverses subdirectories)
        keep_silence_duration: Maximum silence to keep at the end (seconds)
        jitter_enabled: Whether to add random jitter to the cut point
        jitter_range: Tuple of (min_jitter, max_jitter) in seconds
    """
    # Load VAD model
    vad_model = load_silero_vad()
    print("Loaded silero-vad model")

    # Walk through all files
    processed = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                processed += 1

                # Load audio
                y, sr = librosa.load(filepath, sr=None, mono=True)
                audio_length = len(y) / sr

                # Get speech timestamps using VAD
                speech_timestamps = get_speech_timestamps(y, vad_model, sampling_rate=sr, return_seconds=True)

                if not speech_timestamps:
                    print(f"Warning: No speech detected in {file}, skipping")
                    continue

                # Find the actual end time of speech (end of last speech segment)
                speech_end = speech_timestamps[-1]['end']

                # Calculate how much silence is at the end
                silence_at_end = audio_length - speech_end

                # Only trim if there's more silence than we want to keep
                if silence_at_end > keep_silence_duration:
                    # Calculate where to cut (with ±10ms jitter)
                    if jitter_enabled:
                        # ±10ms jitter on the cut point relative to end of speech
                        jitter = random.uniform(-0.010, 0.010)
                        cut_point = speech_end + jitter
                    else:
                        cut_point = speech_end

                    # Ensure we always keep at least keep_silence_duration at the end
                    cut_point = max(keep_silence_duration, min(cut_point, audio_length))

                    # Calculate samples to keep
                    trim_samples = int(cut_point * sr)

                    # Trim from the end
                    y_trimmed = y[:trim_samples]

                    # Save the file
                    sf.write(filepath, y_trimmed, sr, subtype='PCM_16')

                    original_duration = audio_length
                    trimmed_duration = len(y_trimmed) / sr
                    trim_amount = original_duration - trimmed_duration

                    print(f"Processed {processed}: {filepath}")
                    print(f"  Original: {original_duration:.3f}s")
                    print(f"  Trimmed: {trimmed_duration:.3f}s")
                    print(f"  Removed: {trim_amount*1000:.1f}ms ({trim_amount/silence_at_end*100:.1f}% of trailing silence)")
                else:
                    print(f"Skipped {processed}: {filepath}")
                    print(f"  Trailing silence: {silence_at_end*1000:.1f}ms (≤ {keep_silence_duration*1000:.1f}ms threshold)")

    print(f"\nDone! Processed {processed} files.")


if __name__ == '__main__':
    trim_silence()
