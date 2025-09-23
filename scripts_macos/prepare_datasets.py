# scripts_macos/prepare_datasets.py
import os, sys, scipy.io.wavfile, numpy as np
from datasets import Dataset, Audio, load_dataset
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa  # <-- added

# ---------- MIT RIR (robust) ----------
out = Path("mit_rirs")
need_download = (not out.exists()) or (not any(out.glob("*.wav")))
if need_download:
    out.mkdir(parents=True, exist_ok=True)

    def write_wav(dst_dir: Path, name: str, data: np.ndarray, sr: int = 16000):
        scipy.io.wavfile.write(dst_dir / name, sr, (data * 32767).astype(np.int16))

    ok = 0

    # --- Attempt 1: non-streaming decode via datasets.Audio ---
    try:
        ds = load_dataset(
            "davidscripka/MIT_environmental_impulse_responses",
            split="train",
            streaming=False,
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        print("⬇️ MIT RIR (Hugging Face)…")
        for i, row in enumerate(tqdm(ds)):
            arr = row["audio"]["array"]
            if arr is None or len(arr) == 0:
                continue
            name = f"rir_{i:04d}.wav"  # stable numbered filenames
            write_wav(out, name, arr, 16000)
            ok += 1
    except Exception as e:
        print(f"⚠️ Hugging Face MIT RIR failed: {e}")

    # --- Attempt 2: streaming + decode=False ---
    if ok == 0:
        try:
            ds = load_dataset(
                "davidscripka/MIT_environmental_impulse_responses",
                split="train",
                streaming=True,
            )
            ds = ds.cast_column("audio", Audio(decode=False))
            print("⬇️ MIT RIR (streaming, manual decode)…")
            for i, row in enumerate(tqdm(ds)):
                try:
                    src_path = row["audio"]["path"]
                    audio, sr = sf.read(src_path, dtype="float32", always_2d=False)
                    if audio is None or len(audio) == 0:
                        continue
                    name = f"rir_{i:04d}.wav"
                    write_wav(out, name, audio, 16000)
                    ok += 1
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ Streaming/manual decode failed: {e}")

    # --- Attempt 3: direct MIT ZIP (final fallback) ---
    if ok == 0:
        try:
            print("⬇️ MIT RIR (fallback ZIP)…")
            zip_url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
            zip_path = out.parent / "MIT_RIR_Audio.zip"
            if not zip_path.exists():
                os.system(f"wget -q -O {zip_path} {zip_url}")
            os.system(f'unzip -q -o "{zip_path}" -d "{out}"')
            wavs = list(out.rglob("*.wav"))
            ok = len(wavs)
        except Exception as e:
            print(f"⚠️ MIT ZIP fallback failed: {e}")

    print(f"✅ MIT RIR saved: {ok} files")
    if ok == 0:
        print("⚠️ No IRs available; you can temporarily disable RIR augmentation (set 'RIR': 0.0).")
else:
    print("✅ mit_rirs exists and has WAVs; skipping.")

# ---------- AudioSet (agkphysics subset) ----------
audioset_dir = Path("audioset")
audioset_16k = Path("audioset_16k")
audioset_dir.mkdir(exist_ok=True)
audioset_16k.mkdir(exist_ok=True)

links = [
    f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train0{i}.tar"
    for i in range(10)
]
for link in links:
    fname = link.split("/")[-1]
    out_tar = audioset_dir / fname
    if not out_tar.exists():
        print(f"⬇️ {fname}")
        os.system(f"wget -q -O {out_tar} {link}")
        print(f"📦 Extract {fname}")
        os.system(f"tar -xf {out_tar} -C {audioset_dir}")

flacs = list(audioset_dir.glob("**/*.flac"))
print(f"🔎 FLAC files: {len(flacs)}")
corrupt = []
for p in tqdm(flacs, desc="AudioSet→16k WAV"):
    try:
        audio, sr = sf.read(p)
        if audio is None or len(audio) == 0:
            raise ValueError("empty audio")
        scipy.io.wavfile.write(
            audioset_16k / (p.stem + ".wav"),
            16000,
            (audio * 32767).astype(np.int16),
        )
    except Exception as e:
        print(f"⚠️ {p}: {e}")
        corrupt.append(str(p))
if corrupt:
    (audioset_16k / "audioset_corrupted_files.log").write_text("\n".join(corrupt))

# ---------- FMA (xsmall) ----------
fma_zip_dir = Path("fma")
fma_zip_dir.mkdir(exist_ok=True)
fma16 = Path("fma_16k")
fma16.mkdir(exist_ok=True)

zipname = "fma_xs.zip"
zipurl = f"https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/{zipname}"
zipout = fma_zip_dir / zipname
if not zipout.exists():
    os.system(f"wget -q -O {zipout} {zipurl}")
    os.system(f"cd fma && unzip -q {zipname}")

fma_mp3s = list(Path("fma/fma_small").glob("**/*.mp3"))
print(f"🎵 FMA mp3 count: {len(fma_mp3s)}")

corrupt = []
for p in tqdm(fma_mp3s, desc="FMA→16k WAV"):
    try:
        # Robust decode + resample to 16k mono
        y, sr = librosa.load(p, sr=16000, mono=True)
        if y.size == 0:
            raise ValueError("empty audio")
        out_path = fma16 / (p.stem + ".wav")
        scipy.io.wavfile.write(out_path, 16000, (y * 32767).astype(np.int16))
    except Exception as e:
        print(f"⚠️ {p}: {e}")
        corrupt.append(str(p))

if corrupt:
    Path("fma_corrupted_files.log").write_text("\n".join(corrupt))

print("✅ Dataset prep complete!")