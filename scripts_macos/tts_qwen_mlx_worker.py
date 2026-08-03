#!/usr/bin/env python3
"""Persistent Qwen3-TTS worker accelerated by MLX on Apple Silicon."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_audio.audio_io import write as write_audio
from mlx_audio.tts import load


VOICE_DESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit"
VOICE_CLONE_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-6bit"


def read_jsonl(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def chunks(values: list, size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def build_bank(entries: list[dict], output_dir: Path) -> None:
    model = load(VOICE_DESIGN_MODEL)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(entries, start=1):
        mx.random.seed(int(item.get("seed", index)))
        result = next(
            model.generate(
                text=str(item["text"]),
                instruct=str(item["instruct"]),
                lang_code=str(item["language_name"]),
                max_tokens=768,
            )
        )
        write_audio(
            output_dir / f"{item['id']}.wav",
            result.audio,
            result.sample_rate,
        )
        mx.clear_cache()
        if index % 10 == 0 or index == len(entries):
            print(f"Qwen voice design created {index}/{len(entries)}", flush=True)


def generate_direct(entries: list[dict], output_dir: Path, batch_size: int) -> None:
    """Create every corpus clip from a fresh VoiceDesign condition."""

    model = load(VOICE_DESIGN_MODEL)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for batch in chunks(entries, max(1, batch_size)):
        mx.random.seed(int(batch[0].get("seed", completed + 1)))
        results = model.batch_generate(
            texts=[str(item["text"]) for item in batch],
            instructs=[str(item["instruct"]) for item in batch],
            lang_code=str(batch[0]["language_name"]),
            # Qwen emits 12 acoustic frames per second.  Four seconds is a
            # hard ceiling for a wake phrase and prevents decoder rambling.
            max_tokens=48,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.12,
        )
        for result in results:
            item = batch[int(result.sequence_idx)]
            write_audio(
                output_dir / f"{item['id']}.wav",
                result.audio,
                result.sample_rate,
            )
            completed += 1
        mx.clear_cache()
        if completed % 25 == 0 or completed == len(entries):
            print(f"Qwen direct generation created {completed}/{len(entries)}", flush=True)


def generate(
    entries: list[dict], output_dir: Path, batch_size: int
) -> None:
    model = load(VOICE_CLONE_MODEL)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in entries:
        grouped[
            (
                str(item["ref_audio"]),
                str(item["ref_text"]),
                str(item["language_name"]),
            )
        ].append(item)

    completed = 0
    for (ref_audio, ref_text, language_name), group in grouped.items():
        for batch in chunks(group, max(1, batch_size)):
            mx.random.seed(int(batch[0].get("seed", completed + 1)))
            results = model.batch_generate(
                texts=[str(item["text"]) for item in batch],
                ref_audio=ref_audio,
                ref_text=ref_text,
                lang_code=language_name,
                max_tokens=512,
            )
            for result in results:
                item = batch[int(result.sequence_idx)]
                write_audio(
                    output_dir / f"{item['id']}.wav",
                    result.audio,
                    result.sample_rate,
                )
                completed += 1
            mx.clear_cache()
            if completed % 25 == 0 or completed == len(entries):
                print(f"Qwen generated {completed}/{len(entries)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bank", "direct", "generate"), required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    entries = read_jsonl(args.input_jsonl)
    if not entries:
        return 0
    if args.mode == "bank":
        build_bank(entries, args.output_dir)
    elif args.mode == "direct":
        generate_direct(entries, args.output_dir, args.batch_size)
    else:
        generate(entries, args.output_dir, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
