from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from tts_config import parse_omnivoice_catalog

try:
    import trainer_server as trainer
except ModuleNotFoundError:
    trainer = None


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "scripts_macos" / "tts_generate_samples.py"
SPEC = importlib.util.spec_from_file_location("tts_generate_samples", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
generator_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator_module)
QA_PATH = REPO_ROOT / "scripts_macos" / "tts_reference_qa.py"
QA_SPEC = importlib.util.spec_from_file_location("tts_reference_qa", QA_PATH)
assert QA_SPEC is not None and QA_SPEC.loader is not None
qa_module = importlib.util.module_from_spec(QA_SPEC)
QA_SPEC.loader.exec_module(qa_module)


def write_tone(
    path: Path,
    *,
    duration: float = 0.8,
    amplitude: int = 4000,
    frequency: float = 220.0,
) -> None:
    rate = 16000
    samples = array(
        "h",
        (
            int(amplitude * math.sin(2 * math.pi * frequency * index / rate))
            for index in range(int(rate * duration))
        ),
    )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())


class ModernTtsTests(unittest.TestCase):
    def test_direct_generator_uses_one_wake_phrase(self) -> None:
        self.assertEqual(generator_module.reference_text("hey tater"), "hey tater.")
        self.assertEqual(generator_module.reference_text("hey tater!"), "hey tater.")
        self.assertIn("four-provider-direct-corpus", generator_module.GENERATOR_VERSION)
        self.assertIn("safe-limits", generator_module.GENERATOR_VERSION)

    def test_omnivoice_uses_upstream_sampling_defaults(self) -> None:
        self.assertEqual(
            generator_module.omnivoice_stability_args(),
            ["--position_temperature", "5.0", "--class_temperature", "0.0"],
        )

    def test_cli_omnivoice_temp_is_isolated_from_other_engines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            omnivoice_temp = data_dir / "local-socket-temp"
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=1,
                batch_size=4,
                voice_count=2,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            with patch.dict(
                generator_module.os.environ,
                {
                    "TMPDIR": "/Volumes/External/tater-wake-tmp",
                    "MWW_OMNIVOICE_TMPDIR": str(omnivoice_temp),
                },
                clear=False,
            ):
                instance = generator_module.Generator(args)
            selected_temp = generator_module.select_omnivoice_tmpdir(str(omnivoice_temp))
            self.assertEqual(instance.env["TMPDIR"], "/Volumes/External/tater-wake-tmp")
            self.assertEqual(instance.omnivoice_env["TMPDIR"], str(selected_temp))
            self.assertEqual(instance.omnivoice_env["TMP"], str(selected_temp))
            self.assertTrue(selected_temp.is_dir())

    def test_omnivoice_uses_a_hidden_stable_prompt_before_short_clone(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=1,
                batch_size=4,
                voice_count=2,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            instance = generator_module.Generator(args)
            destination = data_dir / "bank"
            destination.mkdir()

            def create_model_outputs(command, *, only_first: bool = False) -> None:
                input_flag = "--test_list" if "--test_list" in command else "--input-jsonl"
                output_flag = "--res_dir" if "--res_dir" in command else "--output-dir"
                input_path = Path(command[command.index(input_flag) + 1])
                output_dir = Path(command[command.index(output_flag) + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                model_entries = [
                    json.loads(line)
                    for line in input_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                for item in model_entries[:1] if only_first else model_entries:
                    write_tone(output_dir / f"{item['id']}.wav")

            with (
                patch.object(instance, "ensure_environment"),
                patch.object(
                    generator_module,
                    "run_with_batch_retry",
                    side_effect=lambda command, _flag, **_kwargs: create_model_outputs(command),
                ) as run_batch,
            ):
                entries = instance._generate_omni_bank(2, 0, destination)

        self.assertEqual(run_batch.call_count, 2)
        self.assertEqual(entries[0]["text"], "hey tater.")
        self.assertEqual(
            entries[0]["ref_text"],
            "In a calm and natural voice, I say hey tater clearly, then continue speaking at an even pace.",
        )
        self.assertEqual(entries[0]["ref_text"].lower().count("hey tater"), 1)
        self.assertIn(".omnivoice-prompts", entries[0]["ref_audio"])
        self.assertEqual(
            entries[0]["voice_description"],
            "automatic random voice",
        )
        self.assertNotIn("instruct", entries[0])
        for call in run_batch.call_args_list:
            command = call.args[0]
            if "--position_temperature" in command:
                self.assertEqual(command[command.index("--position_temperature") + 1], "5.0")
                self.assertEqual(command[command.index("--class_temperature") + 1], "0.0")

    def test_omnivoice_corpus_uses_reference_without_a_second_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=1,
                batch_size=4,
                voice_count=2,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            instance = generator_module.Generator(args)
            entries = instance.make_entries(
                generator_module.ENGINE_OMNIVOICE,
                1,
                [{
                    "id": "omni_ref",
                    "path": "/tmp/short.wav",
                    "ref_text": "hey tater.",
                    "omnivoice_prompt_path": "/tmp/prompt.wav",
                    "omnivoice_prompt_text": "A natural carrier sentence.",
                    "instruct": "female, elderly, low pitch, british accent",
                }],
                data_dir,
            )

        self.assertNotIn("instruct", entries[0])

    def test_omnivoice_corpus_repairs_only_vad_rejected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=2,
                batch_size=4,
                voice_count=2,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            instance = generator_module.Generator(args)
            destination = data_dir / "raw"
            destination.mkdir()
            entries = [{"id": "omni_a", "text": "hey tater."}, {"id": "omni_b", "text": "hey tater."}]
            for entry in entries:
                write_tone(destination / f"{entry['id']}.wav")
            generation_command = [
                "omnivoice",
                "--test_list",
                str(data_dir / "input.jsonl"),
                "--res_dir",
                str(destination),
                "--batch_size",
                "4",
            ]
            qa_calls = 0

            def fake_qa(command, **_kwargs):
                nonlocal qa_calls
                qa_calls += 1
                self.assertIn("--speech-only", command)
                qa_input = Path(command[command.index("--input-jsonl") + 1])
                qa_output = Path(command[command.index("--output-jsonl") + 1])
                candidates = [json.loads(line) for line in qa_input.read_text().splitlines()]
                results = [
                    {
                        "id": item["id"],
                        "accepted": qa_calls > 1 or item["id"] == "omni_a",
                    }
                    for item in candidates
                ]
                qa_output.write_text("".join(json.dumps(item) + "\n" for item in results))

            def fake_retry(command, _flag, **_kwargs):
                retry_input = Path(command[command.index("--test_list") + 1])
                retry_entries = [json.loads(line) for line in retry_input.read_text().splitlines()]
                for entry in retry_entries:
                    write_tone(destination / f"{entry['id']}.wav")

            with (
                patch.object(instance, "_reference_qa_python", return_value=data_dir / "python"),
                patch.object(generator_module, "run", side_effect=fake_qa),
                patch.object(generator_module, "run_with_batch_retry", side_effect=fake_retry) as retry,
            ):
                accepted = instance._repair_generated_corpus(
                    generator_module.ENGINE_OMNIVOICE,
                    entries,
                    destination,
                    generation_command,
                    "",
                    speech_only=True,
                    input_flag="--test_list",
                    batch_flag="--batch_size",
                )
            retry_input = Path(retry.call_args.args[0][retry.call_args.args[0].index("--test_list") + 1])
            retried_ids = [json.loads(line)["id"] for line in retry_input.read_text().splitlines()]

        self.assertEqual([path.name for path in accepted], ["omni_a.wav", "omni_b.wav"])
        self.assertEqual(retried_ids, ["omni_b"])
        self.assertEqual(qa_calls, 2)

    def test_omnivoice_repairs_outputs_missing_from_a_successful_seed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=1,
                batch_size=4,
                voice_count=2,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            instance = generator_module.Generator(args)
            destination = data_dir / "bank"
            destination.mkdir()

            def create_outputs(command, *, only_first: bool = False) -> None:
                input_flag = "--test_list" if "--test_list" in command else "--input-jsonl"
                output_flag = "--res_dir" if "--res_dir" in command else "--output-dir"
                input_path = Path(command[command.index(input_flag) + 1])
                output_dir = Path(command[command.index(output_flag) + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                model_entries = [
                    json.loads(line)
                    for line in input_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                for item in model_entries[:1] if only_first else model_entries:
                    write_tone(output_dir / f"{item['id']}.wav")

            batched_calls = 0

            def fake_batched(command, _flag, **_kwargs):
                nonlocal batched_calls
                batched_calls += 1
                create_outputs(command, only_first=batched_calls == 1)

            with (
                patch.object(instance, "ensure_environment"),
                patch.object(generator_module, "run_with_batch_retry", side_effect=fake_batched),
                patch.object(
                    generator_module,
                    "run",
                    side_effect=lambda command, **_kwargs: create_outputs(command),
                ) as run_single,
            ):
                entries = instance._generate_omni_bank(2, 0, destination)
            retry_command = run_single.call_args.args[0]
            retry_input = Path(retry_command[retry_command.index("--test_list") + 1])
            retried_ids = [json.loads(line)["id"] for line in retry_input.read_text().splitlines()]

        self.assertEqual(len(entries), 2)
        self.assertEqual(batched_calls, 2)
        self.assertEqual(run_single.call_count, 1)
        self.assertEqual(retry_command[retry_command.index("--batch_size") + 1], "1")
        self.assertEqual(retried_ids, ["omni_prompt_0001"])

    def test_reference_semantic_qa_rejects_noise_and_missing_words(self) -> None:
        self.assertTrue(qa_module.transcript_matches_phrase("Hey, Tater.", "hey tater"))
        self.assertTrue(qa_module.transcript_matches_phrase("Hey, gator.", "hey tater"))
        self.assertFalse(qa_module.transcript_matches_phrase("Tater.", "hey tater"))
        self.assertFalse(qa_module.transcript_matches_phrase("Hater.", "hey tater"))
        self.assertFalse(qa_module.transcript_matches_phrase("Thanks for watching!", "hey tater"))
        self.assertFalse(
            qa_module.transcript_matches_phrase("Hey tater. Hey tater.", "hey tater")
        )
        self.assertFalse(qa_module.transcript_matches_phrase("Hey hey Tate", "hey tater"))
        self.assertFalse(qa_module.transcript_matches_phrase("", "hey tater"))
        self.assertEqual(
            qa_module.semantic_rejection_reason("Ehhhhh...", "hey tater", 0.8),
            "decoder_collapse",
        )
        self.assertEqual(
            qa_module.semantic_rejection_reason("Hey tater. Hey tater.", "hey tater", 0.8),
            "repeated_phrase",
        )
        self.assertEqual(
            qa_module.semantic_rejection_reason("Hey hey Tate", "hey tater", 0.8),
            "repeated_phrase",
        )
        self.assertEqual(
            qa_module.semantic_rejection_reason("Hey Taylor", "hey tater", 0.8),
            "phrase_mismatch",
        )

    def test_omnivoice_sample_generation_requires_a_stable_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=1,
                batch_size=4,
                voice_count=2,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            instance = generator_module.Generator(args)
            with self.assertRaisesRegex(RuntimeError, "long-form seed prompt"):
                instance.make_entries(
                    generator_module.ENGINE_OMNIVOICE,
                    1,
                    [{"id": "qwen_ref", "path": "/tmp/qwen.wav", "ref_text": "hey tater."}],
                    data_dir,
                )

    def test_omnivoice_markdown_catalog_parser(self) -> None:
        markdown = """
| # | Language | OmniVoice ID | ISO 639-3 | Duration (h) |
|--:|----------|:------------:|:---------:|:------------:|
| 1 | English | en | eng | 100000.5 |
| 2 | Amdo Tibetan | adx | adx | 56.94 |
"""
        parsed = parse_omnivoice_catalog(markdown)
        self.assertEqual(parsed["en"]["name"], "English")
        self.assertEqual(parsed["adx"]["iso_639_3"], "adx")
        self.assertEqual(parsed["adx"]["duration_hours"], 56.94)

    @unittest.skipIf(trainer is None, "trainer server dependencies are not installed")
    def test_language_catalog_merges_engine_coverage_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    trainer,
                    "_load_omnivoice_catalog",
                    return_value={
                        "en": {"name": "English"},
                        "zu": {"name": "Zulu"},
                    },
                ),
                patch.object(trainer, "_load_piper_catalog", return_value={}),
                patch.object(trainer, "PIPER_ROOT", Path(temp_dir) / "piper"),
                patch.object(trainer, "PIPER_VOICES_DIR", Path(temp_dir) / "voices"),
            ):
                catalog = {item["code"]: item for item in trainer._available_languages()}

        self.assertEqual(catalog["en"]["quality"], "recommended")
        self.assertEqual(catalog["en"]["engines"], ["omnivoice", "qwen3", "moss"])
        self.assertEqual(catalog["zu"]["quality"], "experimental")
        self.assertEqual(catalog["zu"]["engines"], ["omnivoice"])

    @unittest.skipIf(trainer is None, "trainer server dependencies are not installed")
    def test_server_resolves_unavailable_tts_modes_safely(self) -> None:
        languages = [
            {"code": "en", "engines": ["omnivoice", "qwen3", "moss"]},
            {"code": "legacy", "engines": ["piper"]},
        ]
        self.assertEqual(
            trainer._resolve_tts_mode_for_language("piper", "en", languages),
            "modern",
        )
        self.assertEqual(
            trainer._resolve_tts_mode_for_language("modern", "legacy", languages),
            "piper",
        )

    def test_generator_plan_and_piper_discovery_do_not_load_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            output_dir = data_dir / "work" / "wake_word_samples"
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=101,
                batch_size=8,
                voice_count=128,
                data_dir=data_dir,
                output_dir=output_dir,
                ffmpeg="ffmpeg",
                dry_run=True,
                piper_models=[],
            )
            instance = generator_module.Generator(args)
            self.assertEqual(instance.spoken_phrase, "hey tater")
            self.assertEqual(instance.reference_text, "hey tater.")
            self.assertEqual(instance.voice_bank_dir.name, generator_module.phrase_key("hey tater"))
            self.assertEqual(instance.engines(), ["omnivoice", "qwen3", "moss"])
            self.assertEqual(sum(generator_module.distribute_samples(101, instance.engines()).values()), 101)

            model = data_dir / "piper-sample-generator" / "models" / "en_US-libritts_r-medium.pt"
            model.parent.mkdir(parents=True)
            model.touch()
            generator = data_dir / "piper-sample-generator" / "generate_samples.py"
            generator.parent.mkdir(parents=True, exist_ok=True)
            generator.touch()
            piper_python = data_dir / ".venv" / "bin" / "python"
            piper_python.parent.mkdir(parents=True)
            piper_python.touch()
            args.tts_mode = "hybrid"
            self.assertEqual(instance.engines()[-1], "piper")

    def test_voice_descriptions_are_distinct_for_default_bank(self) -> None:
        descriptions = generator_module.qwen_descriptions("English", 128)
        self.assertEqual(len(descriptions), 128)
        self.assertEqual(len(set(descriptions)), 128)

        first_bank = descriptions[:64]
        self.assertEqual(sum(" female speaker " in item for item in first_bank), 32)
        self.assertEqual(sum(" male speaker " in item for item in first_bank), 32)
        for trait in (
            "child",
            "teenager",
            "young adult",
            "middle-aged adult",
            "elderly adult",
            "low pitch",
            "medium pitch",
            "high pitch",
            "calm neutral delivery",
            "bright energetic delivery",
            "soft careful delivery",
            "confident resonant delivery",
            "casual conversational delivery",
            "clear timbre",
            "warm timbre",
            "slightly breathy timbre",
            "crisp timbre",
            "gently rough timbre",
        ):
            self.assertGreaterEqual(sum(trait in item for item in first_bank), 10, trait)

    def test_failed_model_batch_retries_one_item_at_a_time(self) -> None:
        command = ["worker", "--batch-size", "4"]
        with patch.object(
            generator_module,
            "run",
            side_effect=(subprocess.CalledProcessError(1, command), None),
        ) as mocked_run:
            generator_module.run_with_batch_retry(command, "--batch-size")

        self.assertEqual(mocked_run.call_count, 2)
        self.assertEqual(mocked_run.call_args_list[1].args[0], ["worker", "--batch-size", "1"])

    def test_acoustic_qa_accepts_speech_like_pcm_and_rejects_silence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tone = root / "tone.wav"
            silence = root / "silence.wav"
            write_tone(tone)
            write_tone(silence, amplitude=0)
            self.assertTrue(generator_module.valid_sample(tone))
            self.assertTrue(generator_module.valid_reference(tone))
            self.assertFalse(generator_module.valid_sample(silence))

    def test_normalization_skips_a_failed_clip_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=2,
                batch_size=4,
                voice_count=2,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="/opt/homebrew/opt/ffmpeg@7/bin/ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            instance = generator_module.Generator(args)
            paths = [data_dir / "first.wav", data_dir / "second.wav"]
            for path in paths:
                write_tone(path)
            def run_ffmpeg(command, **kwargs):
                if Path(command[command.index("-i") + 1]) == paths[0]:
                    return subprocess.CompletedProcess(
                        args=command,
                        returncode=-6,
                        stdout="",
                        stderr="Assertion best_input >= 0 failed at fftools/ffmpeg_filter.c:2122",
                    )
                write_tone(Path(command[-1]))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch.object(generator_module.subprocess, "run", side_effect=run_ffmpeg) as run:
                normalized = instance.normalize(paths, 0, 2)

            self.assertEqual(normalized, [instance.final_dir / "0.wav"])
            self.assertEqual(instance.normalization_rejections["ffmpeg_failed"], 1)
            self.assertIn("-nostdin", run.call_args_list[0].args[0])
            self.assertEqual(
                run.call_args_list[0].kwargs["timeout"],
                generator_module.FFMPEG_CLIP_TIMEOUT_SECONDS,
            )
        self.assertEqual(run.call_count, 2)

    def test_normalization_skips_a_timed_out_clip_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=2,
                batch_size=4,
                voice_count=2,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            instance = generator_module.Generator(args)
            paths = [data_dir / "first.wav", data_dir / "second.wav"]
            for path in paths:
                write_tone(path)

            def run_ffmpeg(command, **kwargs):
                if Path(command[command.index("-i") + 1]) == paths[0]:
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                write_tone(Path(command[-1]))
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            with patch.object(generator_module.subprocess, "run", side_effect=run_ffmpeg) as run:
                normalized = instance.normalize(paths, 0, 2)

            self.assertEqual(normalized, [instance.final_dir / "0.wav"])
            self.assertEqual(instance.normalization_rejections["ffmpeg_timeout"], 1)
            self.assertEqual(run.call_count, 2)

    def test_normalization_still_fails_if_ffmpeg_cannot_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=1,
                batch_size=1,
                voice_count=1,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="/missing/ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            instance = generator_module.Generator(args)
            source = data_dir / "source.wav"
            write_tone(source)
            with patch.object(
                generator_module.subprocess,
                "run",
                side_effect=OSError("executable not found"),
            ):
                with self.assertRaisesRegex(
                    generator_module.FFmpegRuntimeError,
                    "FFmpeg could not start",
                ):
                    instance.normalize([source], 0, 1)

    def test_provider_safety_gate_rejects_static_and_rambling(self) -> None:
        clean = {
            "duration": 1.2,
            "rms": 0.08,
            "peak": 0.5,
            "clipped_ratio": 0.0,
            "dc_offset": 0.0,
            "spectral_flatness": 0.05,
            "high_frequency_ratio": 0.04,
            "zero_crossing_rate": 0.08,
        }
        self.assertEqual(
            qa_module.acoustic_rejection_reason(clean, 0.7, "omnivoice", 0.4, 2.7),
            "accepted",
        )
        static = {**clean, "spectral_flatness": 0.8}
        self.assertEqual(
            qa_module.acoustic_rejection_reason(static, 0.8, "omnivoice", 0.4, 2.7),
            "static_or_broadband_noise",
        )
        rambling = {**clean, "duration": 3.5}
        self.assertEqual(
            qa_module.acoustic_rejection_reason(rambling, 0.8, "qwen3", 0.4, 2.7),
            "too_long_or_rambling",
        )

    def test_direct_entries_do_not_clone_the_old_voice_bank(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="hybrid",
                samples=12,
                batch_size=4,
                voice_count=128,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            instance = generator_module.Generator(args)
            qwen = instance.make_direct_entries("qwen3", 4, data_dir, [])
            omni = instance.make_direct_entries("omnivoice", 4, data_dir, [])
            refs = [data_dir / f"accepted-{index}.wav" for index in range(4)]
            moss = instance.make_direct_entries("moss", 4, data_dir, refs)

        self.assertTrue(all("ref_audio" not in item for item in qwen + omni))
        self.assertEqual(len({item["instruct"] for item in qwen}), 4)
        self.assertEqual(len({item["seed"] for item in qwen + omni}), 8)
        self.assertEqual([item["ref_audio"] for item in moss], [str(path) for path in refs])

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for normalization")
    def test_orchestrator_produces_exact_normalized_corpus_and_manifest(self) -> None:
        class FakeGenerator(generator_module.Generator):
            generated = 0

            def generate_direct_engine(self, engine, count, reference_paths, prefix=""):
                destination = self.raw_dir / f"{engine}_{prefix or 'main'}"
                destination.mkdir(parents=True, exist_ok=True)
                paths = []
                for index in range(count):
                    path = destination / f"{engine}_{prefix}{index}.wav"
                    write_tone(path, frequency=180 + self.generated)
                    self.generated += 1
                    self.speed_by_path[path.resolve()] = 1.0
                    paths.append(path)
                entries = [
                    {
                        "id": path.stem,
                        "minimum_duration": self.minimum_duration,
                        "maximum_duration": self.maximum_duration,
                    }
                    for path in paths
                ]
                return entries, paths

            def qualify_direct_candidates(self, engine, entries, paths, prefix=""):
                return paths

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            output_dir = data_dir / "work" / "wake_word_samples"
            args = argparse.Namespace(
                phrase="hey tater",
                language="en",
                tts_mode="modern",
                samples=13,
                batch_size=4,
                voice_count=8,
                data_dir=data_dir,
                output_dir=output_dir,
                ffmpeg=shutil.which("ffmpeg"),
                dry_run=False,
                piper_models=[],
            )
            instance = FakeGenerator(args)
            instance.generate()

            self.assertEqual(len(list(output_dir.glob("*.wav"))), 13)
            self.assertTrue((output_dir / ".generation_manifest.json").is_file())
            self.assertTrue(instance.cache_hit())

    def test_apple_training_and_ui_are_wired_for_modern_tts(self) -> None:
        training_script = (REPO_ROOT / "train_microwakeword_macos.sh").read_text(encoding="utf-8")
        self.assertIn("tts_generate_samples.py", training_script)
        self.assertIn('TTS_MODE="${MWW_TTS_MODE:-hybrid}"', training_script)
        self.assertIn('if [[ -n "${REC_VENV_DIR:-}" ]]', training_script)
        self.assertIn('${SUPPORT_DIR}/cli-reference-qa-venv', training_script)
        self.assertIn('export MWW_OMNIVOICE_TMPDIR=', training_script)
        self.assertIn('fallback_temp="/tmp/tw-omni-$(id -u)"', training_script)
        self.assertIn('FFMPEG_BIN="$FFMPEG_PREFIX/bin/ffmpeg"', training_script)
        self.assertIn('"--ffmpeg" "$FFMPEG_BIN"', training_script)
        self.assertIn("ffmpeg_health_check", training_script)
        self.assertIn('select_ffmpeg_formula "ffmpeg"', training_script)
        packaging_lines = training_script.splitlines()
        packaging_start = next(
            index
            for index, line in enumerate(packaging_lines)
            if 'scripts_macos/package_model.py' in line
        )
        for line in packaging_lines[packaging_start : packaging_start + 5]:
            self.assertTrue(line.endswith("\\"), line)
            self.assertFalse(line.endswith("\\\\"), line)
        self.assertIn("--name-by-wake-word", packaging_lines[packaging_start + 5])
        setup_script = (REPO_ROOT / "scripts_macos" / "setup_modern_tts_envs").read_text(encoding="utf-8")
        self.assertIn("mlx-audio[tts]", setup_script)
        self.assertIn("torch==2.8.0", setup_script)
        ui = (REPO_ROOT / "frontend" / "src" / "TrainerApp.vue").read_text(encoding="utf-8")
        store = (REPO_ROOT / "frontend" / "src" / "trainerStore.ts").read_text(encoding="utf-8")
        self.assertIn('v-model="trainer.ttsMode"', ui)
        self.assertIn("tts_mode: trainer.ttsMode", store)
        self.assertIn("OmniVoice", store)

    def test_omnivoice_socket_temp_falls_back_from_long_macos_path(self) -> None:
        long_path = "/var/folders/" + ("x" * 80) + "/T/tater-wake-omnivoice"
        selected = generator_module.select_omnivoice_tmpdir(long_path)
        expected = (Path("/tmp") / f"tw-omni-{os.getuid()}").resolve()

        self.assertEqual(selected, expected)
        self.assertLess(
            len(os.fsencode(str(selected))) + generator_module.OMNIVOICE_SOCKET_SUFFIX_RESERVE,
            generator_module.OMNIVOICE_SOCKET_PATH_LIMIT,
        )

    def test_omnivoice_environment_uses_short_socket_temp_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            args = argparse.Namespace(
                phrase="hey_tater",
                language="en",
                tts_mode="modern",
                samples=1,
                batch_size=1,
                voice_count=1,
                data_dir=data_dir,
                output_dir=data_dir / "work" / "samples",
                ffmpeg="ffmpeg",
                dry_run=False,
                piper_models=[],
            )
            with patch.dict(os.environ, {"MWW_OMNIVOICE_TMPDIR": ""}):
                instance = generator_module.Generator(args)

        selected = instance.omnivoice_env["TMPDIR"]
        self.assertTrue(selected.startswith(("/tmp/", "/private/tmp/")))
        self.assertEqual(instance.omnivoice_env["TMP"], selected)
        self.assertEqual(instance.omnivoice_env["TEMP"], selected)


if __name__ == "__main__":
    unittest.main()
