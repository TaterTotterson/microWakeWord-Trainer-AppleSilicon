#!/usr/bin/env python3
"""Compatibility checks for persistent Apple trainer TTS corpora."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def manifest_matches_requested_corpus(
    manifest_path: Path,
    *,
    phrase: str,
    language: str,
    english_accent: str,
    tts_mode: str,
    samples: int,
    batch_size: int,
    voice_count: int,
) -> bool:
    """Return whether an existing corpus matches user-controlled TTS inputs.

    Older manifests did not record batch size or voice count. Their audio is
    still reusable when all fields they did record match, which lets an app
    update adopt the new stable cache key without regenerating the corpus.
    """
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except Exception:
        return False
    signature = manifest.get("signature")
    if not isinstance(signature, dict):
        return False
    required: dict[str, Any] = {
        "phrase": phrase,
        "language": language,
        "english_accent": english_accent,
        "tts_mode": tts_mode,
        "samples": samples,
    }
    if any(signature.get(key) != value for key, value in required.items()):
        return False
    optional = {"batch_size": batch_size, "voice_count": voice_count}
    return all(
        key not in signature or signature.get(key) == value
        for key, value in optional.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phrase", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--english-accent", required=True)
    parser.add_argument("--tts-mode", required=True)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--voice-count", required=True, type=int)
    args = parser.parse_args()
    return 0 if manifest_matches_requested_corpus(
        args.manifest,
        phrase=args.phrase,
        language=args.language,
        english_accent=args.english_accent,
        tts_mode=args.tts_mode,
        samples=args.samples,
        batch_size=args.batch_size,
        voice_count=args.voice_count,
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
