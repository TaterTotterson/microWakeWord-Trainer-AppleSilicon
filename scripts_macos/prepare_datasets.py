# scripts_macos/prepare_datasets.py
# macOS-friendly dataset prep (no TorchCodec)
# - MIT RIR  -> resample to 16 kHz mono
# - AudioSet -> pinned FLAC .tar revision, resample to 16 kHz mono, skip bad files
# - FMA      -> resample to 16 kHz mono, skip bad files

import os
import subprocess
from pathlib import Path
import numpy as np
import scipy.io.wavfile
import soundfile as sf
import librosa
from tqdm import tqdm

# Optional: keep thread pressure low
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# -----------------------------
# Small helpers
# -----------------------------
def sh(cmd: str) -> int:
    """Run a shell command; return exit code (macOS-safe)."""
    return subprocess.call(cmd, shell=True)

def curl(url: str, dst: Path) -> int:
    """Download via curl -L (follows redirects)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    return sh(f"curl -L --fail -o '{dst}' '{url}'")

def write_wav(dst: Path, data: np.ndarray, sr: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(data, -1.0, 1.0)
    scipy.io.wavfile.write(dst, sr, (x * 32767).astype(np.int16))

# ============================================================
# MIT RIR (ZIP-only, always resample to 16 kHz mono)
# ============================================================
print("=== MIT RIR ===")
rir_out = Path("mit_rirs")
rir_out.mkdir(exist_ok=True)

if not any(rir_out.rglob("*.wav")):
    try:
        print("⬇️ MIT RIR (fallback ZIP)…")
        zip_url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
        zip_path = rir_out.parent / "MIT_RIR_Audio.zip"
        if not zip_path.exists():
            rc = curl(zip_url, zip_path)
            if rc != 0:
                raise RuntimeError("curl download failed for MIT RIR ZIP")

        print("📦 Unzipping…")
        rc = sh(f'unzip -q -o "{zip_path}" -d "{rir_out}"')
        if rc != 0:
            raise RuntimeError("unzip failed for MIT RIR ZIP")

        # Normalize to 16k mono; skip bad files
        normalized = 0
        wavs = list(rir_out.rglob("*.wav"))
        bad = []
        for p in tqdm(wavs, desc="Normalize MIT RIR → 16k mono"):
            try:
                # Robust decode + resample via librosa
                y, _sr = librosa.load(p, sr=16000, mono=True)
                if y is None or y.size == 0:
                    raise ValueError("empty audio")
                write_wav(p, y, 16000)
                normalized += 1
            except Exception as e:
                bad.append(f"{p}:{e}")
        if bad:
            (rir_out / "mit_rir_corrupted_files.log").write_text("\n".join(bad))
        print(f"✅ MIT RIR ready ({normalized} files normalized to 16 kHz, {len(bad)} failed)")
    except Exception as e:
        print(f"❌ MIT RIR ZIP path failed: {e}")
else:
    print("✅ mit_rirs exists; skipping.")

# ============================================================
# AudioSet (pinned FLAC .tar → 16k mono, skip bad files)
# ============================================================
print("\n=== AudioSet subset (pinned FLAC .tar → 16k mono) ===")
audioset_dir = Path("audioset"); audioset_dir.mkdir(exist_ok=True)
audioset_out = Path("audioset_16k"); audioset_out.mkdir(exist_ok=True)

# ✅ skip if already prepared
if any(audioset_out.rglob("*.wav")):
    print("✅ audioset_16k exists; skipping.")
else:
    # Known commits around the conversion period; we probe to find one still serving FLAC tars
    REV_CANDIDATES = [
        "6762f044d1c88619c7f2006486036192128fb07e",
        "0049167e89f259a010c3f070fe3666d9e5242836",
        "ceb9eaaa7844c9ad7351e659c84a572e376ad06d",
        "main",  # last attempt; likely Parquet-only now
    ]
    # Historical layouts we’ve seen
    TAR_PATTERNS = [
        "data/bal_train0{idx}.tar",
        "data/bal_train/bal_train0{idx}.tar",
    ]

    def find_working_rev():
        # Use curl --head --fail to probe
        for rev in REV_CANDIDATES:
            for pat in TAR_PATTERNS:
                probe = f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/{rev}/{pat.format(idx=0)}"
                rc = sh(f"curl -I -L --fail -s '{probe}' > /dev/null")
                if rc == 0:
                    return rev, pat
        return None, None

    rev, pattern = find_working_rev()
    if rev is None:
        raise RuntimeError("Could not locate an AudioSet revision with FLAC tarballs still present on HF.")

    print(f"📌 Using AudioSet revision: {rev}")
    print(f"🗂️ Tar layout pattern: {pattern}")

    # Download & extract bal_train00..09
    for i in range(10):
        rel = pattern.format(idx=i)
        url = f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/{rev}/{rel}"
        fname = rel.split("/")[-1]
        out_tar = audioset_dir / fname
        if not out_tar.exists():
            print(f"⬇️ {fname}")
            rc = curl(url, out_tar)
            if rc != 0:
                print(f"⚠️ Could not fetch {fname} at rev {rev}; continuing.")
                continue
            print(f"📦 Extract {fname}")
            rc = sh(f"tar -xf '{out_tar}' -C '{audioset_dir}'")
            if rc != 0:
                print(f"⚠️ tar extract failed for {fname}; continuing.")

    # Convert all FLAC → 16k mono WAV, skipping bad files
    flacs = list(audioset_dir.rglob("*.flac"))
    print(f"🔎 FLAC files: {len(flacs)}")
    audioset_bad = []
    ok = 0
    for p in tqdm(flacs, desc="AudioSet→WAV (resample 16k mono)"):
        try:
            # librosa handles decode + resample + mono in one step
            y, _ = librosa.load(p, sr=16000, mono=True)
            if y.size == 0:
                raise ValueError("empty audio")
            write_wav(audioset_out / (p.stem + ".wav"), y, 16000)
            ok += 1
        except Exception as e:
            audioset_bad.append(f"{p}:{e}")

    if audioset_bad:
        (audioset_out / "audioset_corrupted_files.log").write_text("\n".join(audioset_bad))
    print(f"✅ AudioSet complete ({ok} ok, {len(audioset_bad)} failed)")

# ============================================================
# FMA xsmall (resample to 16 kHz mono, skip bad files)
# ============================================================
print("\n=== FMA xsmall ===")
fma_zip_dir = Path("fma"); fma_zip_dir.mkdir(exist_ok=True)
fma_out = Path("fma_16k"); fma_out.mkdir(exist_ok=True)

# ✅ skip if already prepared
if any(fma_out.rglob("*.wav")):
    print("✅ fma_16k exists; skipping.")
else:
    zipname = "fma_xs.zip"
    zipurl  = f"https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/{zipname}"
    zipout  = fma_zip_dir / zipname
    if not zipout.exists():
        rc = curl(zipurl, zipout)
        if rc != 0:
            raise RuntimeError("curl download failed for FMA zip")
        rc = sh(f"cd fma && unzip -q '{zipname}'")
        if rc != 0:
            raise RuntimeError("unzip failed for FMA zip")

    mp3s = list(Path("fma/fma_small").rglob("*.mp3"))
    print(f"🎵 FMA mp3 count: {len(mp3s)}")
    fma_bad = []
    ok = 0
    for p in tqdm(mp3s, desc="FMA→WAV (resample 16k mono)"):
        try:
            y, _ = librosa.load(p, sr=16000, mono=True)
            if y.size == 0:
                raise ValueError("empty audio")
            write_wav(fma_out / (p.stem + ".wav"), y, 16000)
            ok += 1
        except Exception as e:
            fma_bad.append(f"{p}:{e}")

    if fma_bad:
        Path("fma_corrupted_files.log").write_text("\n".join(fma_bad))
    print(f"✅ FMA complete ({ok} ok, {len(fma_bad)} failed)")

print("\n✅ Dataset prep complete!")