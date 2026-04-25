#!/usr/bin/env python3
"""Flash a prebuilt ESPHome firmware binary over OTA.

This helper intentionally avoids the full ESPHome compile/run path. The trainer
UI only needs to upload a compiled .bin to a known ESPHome device address.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _add_esphome_source_paths() -> None:
    """Allow using a sibling ESPHome checkout when the package is not installed."""
    candidates = []
    env_path = os.environ.get("ESPHOME_REPO_DIR")
    if env_path:
        candidates.extend(Path(part).expanduser() for part in env_path.split(os.pathsep) if part)

    script_path = Path(__file__).resolve()
    trainer_root = script_path.parents[1]
    siblings_root = script_path.parents[2] if len(script_path.parents) > 2 else trainer_root.parent
    candidates.extend(
        [
            trainer_root / "esphome",
            siblings_root / "esphome",
        ]
    )

    for candidate in candidates:
        if (candidate / "esphome").is_dir():
            sys.path.insert(0, str(candidate))


class LineProgressBar:
    """Console-friendly progress for web log streaming."""

    def __init__(self) -> None:
        self.last_progress: int | None = None

    def update(self, progress: float) -> None:
        progress = max(0.0, min(1.0, float(progress or 0.0)))
        pct = int(progress * 100)
        if pct != 100 and pct % 5 != 0:
            return
        if pct == self.last_progress:
            return
        self.last_progress = pct
        print(f"Uploading firmware: {pct}%", flush=True)

    def done(self) -> None:
        print("Upload transfer complete.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload an ESPHome .bin firmware file over OTA.")
    parser.add_argument("--host", required=True, help="Device IP address or hostname")
    parser.add_argument("--port", type=int, default=3232, help="ESPHome OTA port")
    parser.add_argument("--password", default="", help="Optional OTA password")
    parser.add_argument("firmware", help="Path to a compiled firmware .bin")
    args = parser.parse_args()

    firmware_path = Path(args.firmware).resolve()
    if not firmware_path.exists():
        print(f"Firmware file not found: {firmware_path}", flush=True)
        return 2
    if firmware_path.stat().st_size <= 0:
        print(f"Firmware file is empty: {firmware_path}", flush=True)
        return 2

    _add_esphome_source_paths()
    try:
        from esphome import espota2
    except Exception as exc:
        print("ESPHome Python package is not available.", flush=True)
        print("Install ESPHome in the trainer venv or set ESPHOME_REPO_DIR to an ESPHome checkout.", flush=True)
        print(f"Import error: {exc}", flush=True)
        return 3

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    espota2.ProgressBar = LineProgressBar

    print(f"Preparing OTA upload for {args.host}:{args.port}", flush=True)
    print(f"Firmware: {firmware_path.name} ({firmware_path.stat().st_size:,} bytes)", flush=True)
    exit_code, flashed_host = espota2.run_ota(
        args.host,
        int(args.port),
        args.password or None,
        firmware_path,
    )
    if exit_code == 0:
        print(f"Firmware upload successful{f' to {flashed_host}' if flashed_host else ''}.", flush=True)
    else:
        print("Firmware upload failed.", flush=True)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
