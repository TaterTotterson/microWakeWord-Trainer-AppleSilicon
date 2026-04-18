# scripts_macos/fetch_negatives.py
import requests, zipfile, sys
from pathlib import Path
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

def download(url: str, out: Path):
    r = requests.get(url, stream=True)
    total = int(r.headers.get("content-length", 0))
    with open(out, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out.name) as bar:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))

out_dir = Path("negative_datasets")
out_dir.mkdir(exist_ok=True)

link_root = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
files = ["dinner_party.zip", "dinner_party_eval.zip", "no_speech.zip", "speech.zip"]

for name in files:
    url = link_root + name
    z = out_dir / name
    extract_dir = out_dir / name.removesuffix(".zip")
    if extract_dir.exists():
        print(f"✅ {extract_dir.name} already extracted; skipping.")
        continue
    if not z.exists():
        download(url, z)
    print(f"📦 Extracting {name}…")
    with zipfile.ZipFile(z, "r") as zf:
        zf.extractall(out_dir)
print("✅ Negative datasets ready.")
