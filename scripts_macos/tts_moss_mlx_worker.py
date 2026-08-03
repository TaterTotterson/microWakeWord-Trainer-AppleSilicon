#!/usr/bin/env python3
"""Persistent MOSS-TTS-Nano voice-cloning worker for Apple MLX."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_audio.audio_io import write as write_audio
from mlx_audio.tts import load


MOSS_MODEL = "mlx-community/MOSS-TTS-Nano-100M"


def read_jsonl(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    entries = read_jsonl(args.input_jsonl)
    if not entries:
        return 0

    model = load(MOSS_MODEL)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in entries:
        grouped[str(item["ref_audio"])].append(item)

    completed = 0
    for ref_audio, group in grouped.items():
        prompt_codes = model.encode_reference_audio(
            ref_audio,
            num_quantizers=model.config.n_vq,
        )
        mx.eval(prompt_codes)
        for item in group:
            mx.random.seed(int(item.get("seed", completed + 1)))
            result = next(
                model.generate(
                    text=str(item["text"]),
                    prompt_audio_codes=prompt_codes,
                    # A wake phrase should finish well before this ceiling.
                    # The old 125-token limit allowed occasional long tails.
                    max_tokens=64,
                    do_sample=True,
                    audio_temperature=0.7,
                    audio_top_p=0.9,
                    audio_top_k=25,
                    audio_repetition_penalty=1.3,
                )
            )
            write_audio(
                args.output_dir / f"{item['id']}.wav",
                result.audio,
                result.sample_rate,
            )
            completed += 1
            mx.clear_cache()
            if completed % 10 == 0 or completed == len(entries):
                print(f"MOSS generated {completed}/{len(entries)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
