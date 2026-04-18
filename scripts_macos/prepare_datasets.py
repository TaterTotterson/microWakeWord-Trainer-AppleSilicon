# scripts_macos/prepare_datasets.py
# macOS-friendly dataset prep (no TorchCodec)
# - MIT RIR  -> resample to 16 kHz mono
# - AudioSet -> pinned FLAC .tar revision, resample to 16 kHz mono, skip bad files
# - FMA      -> resample to 16 kHz mono, skip bad files

import os
import subprocess
import sys
import tarfile
import time
import zipfile
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
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

# -----------------------------
# Small helpers
# -----------------------------
def sh(cmd: str) -> int:
    """Run a shell command; return exit code (macOS-safe)."""
    return subprocess.call(cmd, shell=True)

def curl(url: str, dst: Path, attempts: int = 4, backoff_s: float = 2.0) -> int:
    """Download via curl -L with a few retries for transient network failures."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        rc = sh(f"curl -L --fail -o '{dst}' '{url}'")
        if rc == 0 and dst.exists() and dst.stat().st_size > 0:
            return 0
        if dst.exists():
            try:
                dst.unlink()
            except Exception:
                pass
        if attempt < attempts:
            print(f"   Retry {attempt}/{attempts - 1} after download failure...")
            time.sleep(backoff_s * attempt)
    return rc

def download_first_available(urls: list[str], dst: Path, label: str):
    """Try multiple URLs until one succeeds with a non-empty file."""
    last_error = None
    for url in urls:
        print(f"⬇️ {label}")
        rc = curl(url, dst)
        if rc == 0 and dst.exists() and dst.stat().st_size > 0:
            return
        last_error = f"download failed from {url}"
        if dst.exists():
            try:
                dst.unlink()
            except Exception:
                pass
    raise RuntimeError(last_error or f"download failed for {label}")

def ensure_nonempty_download(url: str, dst: Path, label: str):
    """Download a file if missing/empty and fail loudly on bad downloads."""
    if dst.exists() and dst.stat().st_size > 0:
        return
    if dst.exists():
        dst.unlink()
    download_first_available([url], dst, label)

def write_wav(dst: Path, data: np.ndarray, sr: int):
    dst.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(data, -1.0, 1.0)
    scipy.io.wavfile.write(dst, sr, (x * 32767).astype(np.int16))

def extract_zip_with_python(src: Path, dst: Path, label: str):
    """Extract ZIP archives with Python for compatibility with newer ZIP formats."""
    if not src.exists() or src.stat().st_size == 0:
        raise RuntimeError(f"{label} archive is missing or empty: {src}")
    archive_size_gb = src.stat().st_size / (1024 ** 3)
    try:
        with zipfile.ZipFile(src, "r") as zf:
            members = zf.infolist()
            print(f"📦 Extracting {src.name} ({len(members)} entries, {archive_size_gb:.1f} GiB)…")
            for member in tqdm(members, desc=f"Extract {src.name}", unit="file"):
                zf.extract(member, dst)
    except Exception as exc:
        raise RuntimeError(f"Python extraction failed for {label}") from exc

def extract_tar_with_progress(src: Path, dst: Path, label: str, mode: str = "r:*"):
    """Extract tar archives with visible progress so long steps don't look stuck."""
    if not src.exists() or src.stat().st_size == 0:
        raise RuntimeError(f"{label} archive is missing or empty: {src}")
    archive_size_gb = src.stat().st_size / (1024 ** 3)
    try:
        with tarfile.open(src, mode) as tar:
            members = tar.getmembers()
            print(f"📦 Extracting {src.name} ({len(members)} entries, {archive_size_gb:.1f} GiB)…")
            for member in tqdm(members, desc=f"Extract {src.name}", unit="file"):
                tar.extract(member, dst)
    except Exception as exc:
        raise RuntimeError(f"Tar extraction failed for {label}") from exc

def convert_audioset_from_dataset_api(audioset_out: Path):
    """Fallback for current Hugging Face AudioSet layout (Parquet-backed dataset)."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "AudioSet tarballs are unavailable and the Hugging Face datasets API "
            "could not be imported. Ensure the 'datasets' package is installed."
        ) from exc

    print("↪️ AudioSet FLAC tarballs are no longer available; using Hugging Face datasets API instead.")
    try:
        dataset = load_dataset(
            "agkphysics/AudioSet",
            "balanced",
            split="train",
            streaming=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not reach Hugging Face to load AudioSet via the datasets API. "
            "This usually means DNS/network access to huggingface.co failed. "
            "The HF_TOKEN warning is not fatal by itself."
        ) from exc

    audioset_bad = []
    ok = 0
    skipped = 0
    heartbeat_every = 250

    try:
        for idx, sample in enumerate(dataset, start=1):
            try:
                video_id = str(sample.get("video_id") or f"audioset_{idx:06d}")
                outfile = audioset_out / f"{video_id}.wav"
                if outfile.exists():
                    skipped += 1
                    continue

                audio = sample.get("audio") or {}
                y = np.asarray(audio.get("array"))
                sr = int(audio.get("sampling_rate") or 0)
                if y.size == 0 or sr <= 0:
                    raise ValueError("missing decoded audio")
                if y.ndim > 1:
                    y = np.mean(y, axis=-1)
                if sr != 16000:
                    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)
                if y.size == 0:
                    raise ValueError("empty audio")
                write_wav(outfile, y, 16000)
                ok += 1
            except Exception as e:
                audioset_bad.append(f"{sample.get('video_id', idx)}:{e}")

            if idx == 1 or (idx % heartbeat_every) == 0:
                print(f"   AudioSet API progress: {idx} clips processed (ok={ok}, skipped={skipped}, failed={len(audioset_bad)})")
    except Exception as exc:
        raise RuntimeError(
            "AudioSet download via Hugging Face started but then lost network/DNS access. "
            "Please retry when huggingface.co is reachable."
        ) from exc

    if audioset_bad:
        (audioset_out / "audioset_corrupted_files.log").write_text("\n".join(audioset_bad))
    print(f"✅ AudioSet complete via datasets API ({ok} ok, {skipped} skipped, {len(audioset_bad)} failed)")

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

        extract_zip_with_python(zip_path, rir_out, "MIT RIR ZIP")

        # Normalize to 16k mono; skip bad files
        normalized = 0
        print("🔎 Scanning MIT RIR WAV files (please wait)…")
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

    print("🔎 Checking known AudioSet sources (this may take a few seconds)…")
    rev, pattern = find_working_rev()
    if rev is None:
        convert_audioset_from_dataset_api(audioset_out)
    else:
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
                try:
                    extract_tar_with_progress(out_tar, audioset_dir, fname)
                except RuntimeError:
                    print(f"⚠️ tar extract failed for {fname}; continuing.")

        # Convert all FLAC → 16k mono WAV, skipping bad files
        print("🔎 Scanning extracted AudioSet FLAC files (please wait)…")
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

        print("⏳ Finalizing AudioSet output (writing logs and checking results)…")
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
    zipname = "fma_small.zip"
    zipurls = [
        "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
        "https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip",
    ]
    zipout  = fma_zip_dir / zipname
    extracted_fma_dir = fma_zip_dir / "fma_small"
    if not zipout.exists():
        download_first_available(zipurls, zipout, zipname)
    if not extracted_fma_dir.exists() or not any(extracted_fma_dir.rglob("*.mp3")):
        extract_zip_with_python(zipout, fma_zip_dir, "FMA zip")

    print("🔎 Scanning extracted FMA audio files (please wait)…")
    mp3s = list(extracted_fma_dir.rglob("*.mp3"))
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

    print("⏳ Finalizing FMA output (writing logs and checking results)…")
    if fma_bad:
        Path("fma_corrupted_files.log").write_text("\n".join(fma_bad))
    print(f"✅ FMA complete ({ok} ok, {len(fma_bad)} failed)")

# ============================================================
# WHAM_noise (resample to 16 kHz mono, skip bad files)
# ============================================================
print("\n=== WHAM noise ===")
wham_zip_dir = Path("wham"); wham_zip_dir.mkdir(exist_ok=True)
wham_out = Path("wham_16k"); wham_out.mkdir(exist_ok=True)

if any(wham_out.rglob("*.wav")):
    print("✅ wham_16k exists; skipping.")
else:
    zipname = "wham_noise.zip"
    zipurl = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
    zipout = wham_zip_dir / zipname

    if not zipout.exists():
        print(f"⬇️ {zipname}")
        rc = sh(f"wget -q -O '{zipout}' '{zipurl}'")
        if rc != 0:
            raise RuntimeError("wget failed for WHAM noise zip")
    extracted_wham_dir = wham_zip_dir / "wham_noise"
    if not extracted_wham_dir.exists() or not any(extracted_wham_dir.rglob("*.wav")):
        extract_zip_with_python(zipout, wham_zip_dir, "WHAM noise zip")

    # Find all wav files in the wham directory
    print("🔎 Scanning extracted WHAM WAV files (please wait)…")
    wavs = list(wham_zip_dir.rglob("*.wav"))
    print(f"WHAM WAV count: {len(wavs)}")

    corrupt = []
    for p in tqdm(wavs, desc="WHAM→16k WAV"):
        try:
            y, _ = librosa.load(p, sr=16000, mono=True)
            if y.size == 0:
                raise ValueError("empty audio")
            write_wav(wham_out / (p.stem + ".wav"), y, 16000)
        except Exception as e:
            corrupt.append(f"{p}:{e}")

    print("⏳ Finalizing WHAM output (writing logs and checking results)…")
    if corrupt:
        Path("wham_corrupted_files.log").write_text("\n".join(corrupt))
    print(f"✅ WHAM complete (handled {len(corrupt)} corrupt files)")

# ============================================================
# CHiME-Home (resample to 16 kHz mono, skip bad files)
# ============================================================
print("\n=== CHiME-Home ===")
chime_tar_dir = Path("chime"); chime_tar_dir.mkdir(exist_ok=True)
chime_out = Path("chime_16k"); chime_out.mkdir(exist_ok=True)

if any(chime_out.rglob("*.wav")):
    print("✅ chime_16k exists; skipping.")
else:
    tar_filename = "chime_home.tar.gz"
    tar_url = "https://archive.org/download/chime-home/chime_home.tar.gz"
    tar_path = chime_tar_dir / tar_filename

    ensure_nonempty_download(tar_url, tar_path, tar_filename)

    try:
        extract_tar_with_progress(tar_path, chime_tar_dir, tar_filename, "r:gz")
    except (RuntimeError, tarfile.ReadError) as exc:
        try:
            tar_path.unlink()
        except Exception:
            pass
        raise RuntimeError(
            f"{tar_filename} was empty or invalid. It has been removed so the next run can re-download it."
        ) from exc

    # Remove the tar file to save space
    tar_path.unlink()

    # Find all wav files in the chime directory (*.48kHz.wav format)
    print("🔎 Scanning extracted CHiME WAV files (please wait)…")
    wavs = list(chime_tar_dir.rglob("*.48kHz.wav"))
    print(f"CHiME WAV count: {len(wavs)}")

    corrupt = []
    for p in tqdm(wavs, desc="CHiME→16k WAV"):
        try:
            y, _ = librosa.load(p, sr=16000, mono=True)
            if y.size == 0:
                raise ValueError("empty audio")
            write_wav(chime_out / (p.stem + ".wav"), y, 16000)
        except Exception as e:
            corrupt.append(f"{p}:{e}")

    print("⏳ Finalizing CHiME output (writing logs and checking results)…")
    if corrupt:
        Path("chime_corrupted_files.log").write_text("\n".join(corrupt))
    print(f"✅ CHiME complete (handled {len(corrupt)} corrupt files)")

print("\n✅ Dataset prep complete!")
