#!/usr/bin/env python3

# trainer_server.py
import contextlib
import fcntl
import gc
import hashlib
import io
import os
import queue
import re
import json
import secrets
import signal
import socket
import stat as stat_module
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import wave
from array import array
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from math import isfinite, log10
from pathlib import Path
from typing import Dict, Any, List, Callable
from urllib.parse import quote
from urllib.error import HTTPError
from urllib.request import Request as URLRequest, urlopen

from fastapi import FastAPI, UploadFile, File, Form, Header, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent

from tts_config import (
    COMMON_OMNIVOICE_LANGUAGES,
    DEFAULT_ENGLISH_ACCENT,
    DEFAULT_TTS_MODE,
    ENGINE_MOSS,
    ENGINE_OMNIVOICE,
    ENGINE_PIPER,
    ENGINE_QWEN3,
    MOSS_LANGUAGES,
    OMNIVOICE_LANGUAGE_ALIASES,
    QWEN_LANGUAGES,
    english_accent_options,
    normalize_english_accent,
    normalize_tts_mode,
    parse_omnivoice_catalog,
    quality_for_engines,
)

SUPPORT_DIR = Path(
    os.environ.get(
        "WAKEWORD_TRAINER_SUPPORT_DIR",
        str(Path.home() / ".taterwakewordtrainer"),
    )
).resolve()
DATA_DIR = Path(
    os.environ.get(
        "WAKEWORD_TRAINER_DATA_DIR",
        str(SUPPORT_DIR / "app" / "current"),
    )
).resolve()
STATIC_DIR = Path(os.environ.get("STATIC_DIR", str(ROOT_DIR / "static"))).resolve()
PERSONAL_DIR = Path(os.environ.get("PERSONAL_DIR", str(DATA_DIR / "personal_samples"))).resolve()
CAPTURED_DIR = Path(os.environ.get("CAPTURED_DIR", str(DATA_DIR / "captured_audio"))).resolve()
NEGATIVE_DIR = Path(os.environ.get("NEGATIVE_DIR", str(DATA_DIR / "negative_samples"))).resolve()
TRIM_HISTORY_DIR = Path(os.environ.get("TRIM_HISTORY_DIR", str(DATA_DIR / "trim_history"))).resolve()
TRIM_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
TRAINED_WAKE_WORDS_DIR = Path(
    os.environ.get("TRAINED_WAKE_WORDS_DIR", str(DATA_DIR / "trained_wake_words"))
).resolve()
AUTO_TRAIN_CONFIG_FILE = Path(
    os.environ.get("AUTO_TRAIN_CONFIG_FILE", str(DATA_DIR / "auto_train_config.json"))
).resolve()
AUTO_TRAIN_STATE_FILE = Path(
    os.environ.get("AUTO_TRAIN_STATE_FILE", str(DATA_DIR / "auto_train_state.json"))
).resolve()
AUTO_TRAIN_MODEL_DIR = Path(
    os.environ.get("AUTO_TRAIN_MODEL_DIR", str(DATA_DIR / "auto_train_models"))
).resolve()
TRAINING_LOCK_FILE = Path(
    os.environ.get(
        "WAKEWORD_TRAINER_TRAINING_LOCK_FILE",
        str(SUPPORT_DIR / "training.lock"),
    )
).resolve()
TRAIN_SCRIPT = str(
    Path(
        os.environ.get(
            "TRAIN_SCRIPT",
            str(ROOT_DIR / "train_microwakeword_macos.sh"),
        )
    ).resolve()
)
PIPER_ROOT = DATA_DIR / "piper-sample-generator"
PIPER_VOICES_DIR = PIPER_ROOT / "voices"
PIPER_VOICES_INDEX_URL = os.environ.get(
    "PIPER_VOICES_INDEX_URL",
    "https://huggingface.co/rhasspy/piper-voices/raw/main/voices.json",
)
PIPER_VOICES_ROOT_URL = os.environ.get(
    "PIPER_VOICES_ROOT_URL",
    "https://huggingface.co/rhasspy/piper-voices/resolve/main",
)
PIPER_CATALOG_CACHE_TTL_SECONDS = int(os.environ.get("PIPER_CATALOG_CACHE_TTL_SECONDS", "900"))
PIPER_CATALOG_CACHE_FILE = Path(
    os.environ.get(
        "PIPER_CATALOG_CACHE_FILE",
        str(DATA_DIR / ".cache" / "piper_voices_catalog.json"),
    )
).resolve()
OMNIVOICE_LANGUAGES_URL = os.environ.get(
    "OMNIVOICE_LANGUAGES_URL",
    "https://raw.githubusercontent.com/k2-fsa/OmniVoice/main/docs/languages.md",
)
OMNIVOICE_CATALOG_CACHE_TTL_SECONDS = int(
    os.environ.get("OMNIVOICE_CATALOG_CACHE_TTL_SECONDS", "86400")
)
OMNIVOICE_CATALOG_CACHE_FILE = Path(
    os.environ.get(
        "OMNIVOICE_CATALOG_CACHE_FILE",
        str(DATA_DIR / ".cache" / "omnivoice_languages.json"),
    )
).resolve()
DEFAULT_LANGUAGE = os.environ.get("MWW_LANGUAGE", "en")
DEFAULT_SERVER_TTS_MODE = normalize_tts_mode(
    os.environ.get("MWW_TTS_MODE", DEFAULT_TTS_MODE)
)
DEFAULT_SERVER_ENGLISH_ACCENT = normalize_english_accent(
    os.environ.get("MWW_ENGLISH_ACCENT", DEFAULT_ENGLISH_ACCENT),
    DEFAULT_LANGUAGE,
)
DEFAULT_TTS_VOICE_COUNT = max(1, int(os.environ.get("MWW_TTS_VOICE_COUNT", "128")))

TAKES_PER_SPEAKER_DEFAULT = int(os.environ.get("REC_TAKES_PER_SPEAKER", "10"))
SPEAKERS_TOTAL_DEFAULT = int(os.environ.get("REC_SPEAKERS_TOTAL", "1"))
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH_BYTES = 2
CAPTURE_GAIN_PROFILE = "capture_rms_v1"
STT_ENGINE_FASTER_WHISPER = "faster_whisper"
STT_ENGINE_PARAKEET_ONNX = "parakeet_onnx"
STT_ENGINE_MLX_WHISPER = "mlx_whisper"
SUPPORTED_STT_ENGINES = {
    STT_ENGINE_FASTER_WHISPER,
    STT_ENGINE_PARAKEET_ONNX,
    STT_ENGINE_MLX_WHISPER,
}
DEFAULT_STT_ENGINE = os.environ.get(
    "AUTO_TRAIN_STT_ENGINE",
    STT_ENGINE_FASTER_WHISPER,
).strip().lower().replace("-", "_")
if DEFAULT_STT_ENGINE not in SUPPORTED_STT_ENGINES:
    DEFAULT_STT_ENGINE = STT_ENGINE_FASTER_WHISPER
DEFAULT_FASTER_WHISPER_EN_MODEL = os.environ.get(
    "AUTO_TRAIN_FASTER_WHISPER_EN_MODEL",
    "small.en",
)
DEFAULT_FASTER_WHISPER_MULTILINGUAL_MODEL = os.environ.get(
    "AUTO_TRAIN_FASTER_WHISPER_MULTILINGUAL_MODEL",
    "small",
)
DEFAULT_MLX_WHISPER_EN_MODEL = os.environ.get(
    "AUTO_TRAIN_MLX_WHISPER_EN_MODEL",
    "mlx-community/whisper-base.en-mlx",
)
DEFAULT_MLX_WHISPER_MULTILINGUAL_MODEL = os.environ.get(
    "AUTO_TRAIN_MLX_WHISPER_MULTILINGUAL_MODEL",
    "mlx-community/whisper-base-mlx",
)
DEFAULT_PARAKEET_ONNX_MODEL = os.environ.get(
    "AUTO_TRAIN_PARAKEET_ONNX_MODEL",
    "nemo-parakeet-tdt-0.6b-v3",
)
DEFAULT_PARAKEET_ONNX_REPO = os.environ.get(
    "AUTO_TRAIN_PARAKEET_ONNX_REPO",
    "istupakov/parakeet-tdt-0.6b-v3-onnx",
)
DEFAULT_PARAKEET_ONNX_QUANTIZATION = "int8"
WAKE_PHRASE_GUIDANCE_MIN_SIMILARITY = 0.68

AUTO_TRAIN_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "wake_phrase": "",
    "language": DEFAULT_LANGUAGE,
    "english_accent": DEFAULT_SERVER_ENGLISH_ACCENT,
    "stt_engine": DEFAULT_STT_ENGINE,
    "minimum_transcript_chars": 2,
    "delete_confirmed_wakes": False,
    "promote_close_misses": False,
    "schedule_hours": 24,
    "minimum_new_negatives": 3,
    "advertised_base_url": "",
    "tater_url": "http://127.0.0.1:8501",
    "tater_link_token": "",
    "tater_link_id": "",
    "tater_linked_at": "",
    "tater_link_tater_name": "",
    "notify_satellites": True,
}

AUTO_TRAIN_DEFAULT_STATE: Dict[str, Any] = {
    "pending_negative_count": 0,
    "next_run_at": "",
    "last_review_at": "",
    "last_review_file": "",
    "last_review_transcript": "",
    "last_review_result": "",
    "last_review_error": "",
    "last_stt_engine": "",
    "last_stt_model": "",
    "last_stt_device": "",
    "last_stt_compute_type": "",
    "last_train_started_at": "",
    "last_train_finished_at": "",
    "last_train_exit_code": None,
    "last_notify_at": "",
    "last_notify_count": None,
    "last_notify_error": "",
}

app = FastAPI(title="microWakeWord Personal Samples")

# Serve static UI
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def safe_name(raw: str) -> str:
    s = unicodedata.normalize("NFKC", raw or "").strip().lower()
    digest_source = s
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"^_+|_+$", "", s)
    if s:
        return s
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:8]
    return f"wakeword_{digest}"


# -------------------- In-memory session state --------------------
STATE: Dict[str, Any] = {
    "raw_phrase": None,
    "safe_word": None,
    "language": DEFAULT_LANGUAGE,
    "english_accent": DEFAULT_SERVER_ENGLISH_ACCENT,
    "tts_mode": DEFAULT_SERVER_TTS_MODE,

    # multi-speaker
    "speakers_total": SPEAKERS_TOTAL_DEFAULT,
    "takes_per_speaker": TAKES_PER_SPEAKER_DEFAULT,

    # recording progress
    "takes_received": 0,   # total across all speakers
    "takes": [],           # list of saved filenames

    "training": {
        "running": False,
        "exit_code": None,
        "log_lines": [],
        "log_path": None,
        "safe_word": None,
    },
}

STATE_LOCK = threading.Lock()
SAMPLES_LOCK = threading.Lock()
DATA_MANAGEMENT_LOCK = threading.RLock()
PIPER_CATALOG_LOCK = threading.Lock()
OMNIVOICE_CATALOG_LOCK = threading.Lock()
AUTO_TRAIN_LOCK = threading.RLock()
AUTO_TRAIN_WAKE_EVENT = threading.Event()
AUTO_TRAIN_STOP_EVENT = threading.Event()
AUTO_TRAIN_REVIEW_QUEUE: queue.Queue[str] = queue.Queue()
AUTO_TRAIN_QUEUED_FILES: set[str] = set()
AUTO_TRAIN_WORKER: threading.Thread | None = None
TRAINING_RUNTIME_LOCK = threading.RLock()
TRAINING_SHUTDOWN_EVENT = threading.Event()
TRAINING_STOP_EVENT = threading.Event()
TRAINING_PROCESS: subprocess.Popen | None = None
TRAINING_THREAD: threading.Thread | None = None
AUTO_TRAIN_RUNTIME: Dict[str, Any] = {
    "review_running": False,
    "review_file": "",
    "scheduler_running": False,
    "training_pending_consumed": 0,
}
LAN_ADDRESS_CACHE: Dict[str, Any] = {"value": "", "fetched_at": 0.0}
FASTER_WHISPER_MODEL_LOCK = threading.RLock()
FASTER_WHISPER_MODEL_CACHE: Dict[tuple[str, str, str], Any] = {}
FASTER_WHISPER_TRANSCRIBE_LOCK = threading.RLock()
MLX_WHISPER_TRANSCRIBE_LOCK = threading.RLock()
PARAKEET_ONNX_MODEL_LOCK = threading.RLock()
PARAKEET_ONNX_MODEL_CACHE: Dict[tuple[str, str, tuple[str, ...]], Any] = {}
PARAKEET_ONNX_TRANSCRIBE_LOCK = threading.RLock()
PIPER_CATALOG_CACHE: Dict[str, Any] = {
    "fetched_at": 0.0,
    "entries": None,
}
OMNIVOICE_CATALOG_CACHE: Dict[str, Any] = {
    "fetched_at": 0.0,
    "entries": None,
}


def _managed_data_registry() -> List[Dict[str, Any]]:
    """Return the exact trainer-owned paths that the Data tab may remove."""
    rebuild = "The trainer will rebuild this automatically when it is needed again."
    redownload = "The trainer will download this again when it is needed."
    irreplaceable = "These recordings are not generated and cannot be restored automatically."
    return [
        {"id": "personal_samples", "label": "Personal positive samples", "category": "Recordings", "description": "User recordings and imported positive wake-word clips.", "paths": [PERSONAL_DIR], "rebuild_note": irreplaceable},
        {"id": "negative_samples", "label": "Reviewed negative samples", "category": "Recordings", "description": "Reviewed false wakes and other hard-negative recordings.", "paths": [NEGATIVE_DIR], "rebuild_note": irreplaceable},
        {"id": "captured_audio", "label": "Captured-audio inbox", "category": "Recordings", "description": "Unreviewed audio received from Tater satellites.", "paths": [CAPTURED_DIR], "rebuild_note": irreplaceable},
        {"id": "trim_history", "label": "Audio trim history", "category": "Recordings", "description": "Original audio retained so sample trims can be reverted.", "paths": [TRIM_HISTORY_DIR], "rebuild_note": "Deleting this removes the ability to revert existing trims."},

        {"id": "generated_samples", "label": "Generated wake-word samples", "category": "Generated training data", "description": "The direct TTS corpus used for the current wake word.", "paths": [DATA_DIR / "generated_samples"], "rebuild_note": rebuild},
        {"id": "generation_staging", "label": "TTS generation staging", "category": "Generated training data", "description": "Raw, quality-check, and partial files from an in-progress or interrupted generation.", "paths": [DATA_DIR / ".wake_word_samples.build"], "rebuild_note": rebuild},
        {"id": "generated_features", "label": "Generated augmented features", "category": "Generated training data", "description": "Augmented model features produced from generated speech.", "paths": [DATA_DIR / "generated_augmented_features"], "rebuild_note": rebuild},
        {"id": "personal_features", "label": "Personal augmented features", "category": "Generated training data", "description": "Training features derived from personal positive samples.", "paths": [DATA_DIR / "personal_augmented_features"], "rebuild_note": rebuild},
        {"id": "reviewed_negative_features", "label": "Reviewed-negative features", "category": "Generated training data", "description": "Training features derived from reviewed false wakes.", "paths": [DATA_DIR / "reviewed_negative_features"], "rebuild_note": rebuild},

        {"id": "negative_speech", "label": "Speech negatives", "category": "Downloaded training datasets", "description": "Stock non-wake speech features used to reduce false activations.", "paths": [DATA_DIR / "negative_datasets" / "speech"], "rebuild_note": redownload},
        {"id": "negative_dinner_party", "label": "Dinner-party negatives", "category": "Downloaded training datasets", "description": "Overlapping conversational noise used during training.", "paths": [DATA_DIR / "negative_datasets" / "dinner_party"], "rebuild_note": redownload},
        {"id": "negative_no_speech", "label": "No-speech negatives", "category": "Downloaded training datasets", "description": "Ambient non-speech features used during training.", "paths": [DATA_DIR / "negative_datasets" / "no_speech"], "rebuild_note": redownload},
        {"id": "negative_dinner_eval", "label": "Dinner-party evaluation set", "category": "Downloaded training datasets", "description": "Held-out conversational audio used to evaluate false activations.", "paths": [DATA_DIR / "negative_datasets" / "dinner_party_eval"], "rebuild_note": redownload},
        {"id": "mit_rirs", "label": "MIT room impulse responses", "category": "Downloaded training datasets", "description": "Room acoustics used to make generated voices more realistic.", "paths": [DATA_DIR / "mit_rirs"], "rebuild_note": redownload},
        {"id": "audioset_source", "label": "AudioSet source download", "category": "Downloaded training datasets", "description": "Original downloaded AudioSet material retained for preparation.", "paths": [DATA_DIR / "audioset"], "rebuild_note": redownload},
        {"id": "audioset_16k", "label": "AudioSet 16 kHz training audio", "category": "Downloaded training datasets", "description": "Prepared AudioSet audio used for augmentation.", "paths": [DATA_DIR / "audioset_16k"], "rebuild_note": redownload},
        {"id": "fma_source", "label": "FMA source download", "category": "Downloaded training datasets", "description": "Original downloaded Free Music Archive material.", "paths": [DATA_DIR / "fma"], "rebuild_note": redownload},
        {"id": "fma_16k", "label": "FMA 16 kHz training audio", "category": "Downloaded training datasets", "description": "Prepared music audio used for augmentation.", "paths": [DATA_DIR / "fma_16k"], "rebuild_note": redownload},
        {"id": "wham_source", "label": "WHAM! source download", "category": "Downloaded training datasets", "description": "Original downloaded WHAM! background-noise material.", "paths": [DATA_DIR / "wham"], "rebuild_note": redownload},
        {"id": "wham_16k", "label": "WHAM! 16 kHz training audio", "category": "Downloaded training datasets", "description": "Prepared WHAM! noise used for augmentation.", "paths": [DATA_DIR / "wham_16k"], "rebuild_note": redownload},
        {"id": "chime_source", "label": "CHiME source download", "category": "Downloaded training datasets", "description": "Original downloaded CHiME household-noise material.", "paths": [DATA_DIR / "chime"], "rebuild_note": redownload},
        {"id": "chime_16k", "label": "CHiME 16 kHz training audio", "category": "Downloaded training datasets", "description": "Prepared CHiME noise used for augmentation.", "paths": [DATA_DIR / "chime_16k"], "rebuild_note": redownload},

        {"id": "omnivoice_environment", "label": "OmniVoice engine", "category": "Voice and speech models", "description": "The isolated OmniVoice runtime and installed packages.", "paths": [DATA_DIR / "tts-envs" / "omnivoice"], "rebuild_note": redownload},
        {"id": "qwen_environment", "label": "Qwen3-TTS engine", "category": "Voice and speech models", "description": "The isolated Qwen3-TTS runtime and installed packages.", "paths": [DATA_DIR / "tts-envs" / "qwen3"], "rebuild_note": redownload},
        {"id": "moss_environment", "label": "MOSS-TTS engine", "category": "Voice and speech models", "description": "The isolated MOSS-TTS runtime and installed packages.", "paths": [DATA_DIR / "tts-envs" / "moss"], "rebuild_note": redownload},
        {"id": "tts_model_cache", "label": "TTS model downloads", "category": "Voice and speech models", "description": "Hugging Face model weights shared by the modern TTS providers.", "paths": [DATA_DIR / ".cache" / "huggingface"], "rebuild_note": redownload},
        {"id": "piper_models", "label": "Piper voice models", "category": "Voice and speech models", "description": "Downloaded Piper model weights used by hybrid and legacy generation.", "paths": [PIPER_ROOT / "models"], "rebuild_note": redownload},
        {"id": "piper_voices", "label": "Additional Piper voices", "category": "Voice and speech models", "description": "Language-specific Piper voices selected by the trainer.", "paths": [PIPER_VOICES_DIR], "rebuild_note": redownload},
        {"id": "stt_models", "label": "Auto-training STT models", "category": "Voice and speech models", "description": "Whisper, Parakeet, and other speech-recognition model downloads.", "paths": [AUTO_TRAIN_MODEL_DIR], "rebuild_note": redownload},
        {"id": "provider_catalogs", "label": "Voice-provider catalogs", "category": "Voice and speech models", "description": "Cached OmniVoice language and Piper voice listings.", "paths": [OMNIVOICE_CATALOG_CACHE_FILE, PIPER_CATALOG_CACHE_FILE], "rebuild_note": redownload},
        {"id": "voice_bank", "label": "Legacy voice-bank references", "category": "Voice and speech models", "description": "Reference clips left by older voice-bank generation runs.", "paths": [DATA_DIR / "voice-bank"], "rebuild_note": rebuild},

        {"id": "training_workspace", "label": "Model training workspace", "category": "Training results", "description": "Checkpoints, logs, and intermediate files from the latest model run.", "paths": [DATA_DIR / "trained_models"], "rebuild_note": rebuild},
        {"id": "published_models", "label": "Published wake-word models", "category": "Training results", "description": "Finished TFLite models and JSON packages shown in Wake Words.", "paths": [TRAINED_WAKE_WORDS_DIR], "rebuild_note": "Tater links to these files will stop working. Train again to recreate them."},
        {"id": "training_log", "label": "Training console log", "category": "Training results", "description": "Saved console output from the most recent training run.", "paths": [DATA_DIR / "recorder_training.log"], "rebuild_note": "The deleted history cannot be restored; the next run creates a new log."},

        {"id": "archived_voice_banks", "label": "Archived legacy voice banks", "category": "Legacy and quarantined data", "description": "Older reference banks retained outside the active training workspace.", "paths": [SUPPORT_DIR / "voice-bank-archive"], "rebuild_note": "These archived references cannot be restored automatically."},
        {"id": "quarantined_audio", "label": "Quarantined audio", "category": "Legacy and quarantined data", "description": "Rejected or diagnostic clips intentionally kept out of active training.", "paths": [SUPPORT_DIR / "quarantine"], "rebuild_note": "These diagnostic files cannot be restored automatically."},
    ]


def _managed_data_location(paths: List[Path]) -> str:
    locations: List[str] = []
    for path in paths:
        try:
            locations.append(str(path.relative_to(DATA_DIR)))
        except ValueError:
            locations.append(path.name)
    return ", ".join(locations)


def _managed_path_usage(path: Path) -> tuple[int, int]:
    """Return allocated bytes and file count without following symbolic links."""
    if not os.path.lexists(path):
        return 0, 0
    total_bytes = 0
    file_count = 0
    stack = [os.fspath(path)]
    seen: set[tuple[int, int]] = set()
    while stack:
        current = stack.pop()
        try:
            stat = os.lstat(current)
        except OSError:
            continue
        if stat_module.S_ISLNK(stat.st_mode) or not stat_module.S_ISDIR(stat.st_mode):
            inode = (int(stat.st_dev), int(stat.st_ino))
            if inode in seen:
                continue
            seen.add(inode)
            allocated = int(getattr(stat, "st_blocks", 0) or 0) * 512
            total_bytes += allocated or int(stat.st_size)
            file_count += 1
            continue
        try:
            with os.scandir(current) as entries:
                stack.extend(entry.path for entry in entries)
        except OSError:
            continue
    return total_bytes, file_count


def _managed_data_payload() -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    total_size = 0
    total_files = 0
    with DATA_MANAGEMENT_LOCK:
        for definition in _managed_data_registry():
            paths = [Path(path) for path in definition["paths"]]
            usages = [_managed_path_usage(path) for path in paths]
            size_bytes = sum(size for size, _ in usages)
            file_count = sum(count for _, count in usages)
            total_size += size_bytes
            total_files += file_count
            items.append({
                **{key: value for key, value in definition.items() if key != "paths"},
                "location": _managed_data_location(paths),
                "size_bytes": size_bytes,
                "file_count": file_count,
                "exists": any(os.path.lexists(path) for path in paths),
            })
    return {
        "ok": True,
        "items": items,
        "total_size_bytes": total_size,
        "total_file_count": total_files,
    }


def _remove_managed_path(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _clear_auto_review_queue() -> None:
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_QUEUED_FILES.clear()
    while True:
        try:
            AUTO_TRAIN_REVIEW_QUEUE.get_nowait()
        except queue.Empty:
            break
        else:
            AUTO_TRAIN_REVIEW_QUEUE.task_done()


def _delete_managed_data_item(item_id: str) -> Dict[str, Any]:
    definitions = {item["id"]: item for item in _managed_data_registry()}
    definition = definitions.get(str(item_id or ""))
    if definition is None:
        raise KeyError("Unknown managed data item.")
    paths = [Path(path) for path in definition["paths"]]
    with DATA_MANAGEMENT_LOCK:
        with STATE_LOCK:
            if STATE["training"]["running"]:
                raise RuntimeError("Stop training before deleting trainer data.")
        with AUTO_TRAIN_LOCK:
            if AUTO_TRAIN_RUNTIME.get("review_running"):
                raise RuntimeError("Wait for the current automatic audio review to finish before deleting data.")
        previous_size = sum(_managed_path_usage(path)[0] for path in paths)
        for path in paths:
            _remove_managed_path(path)
        if item_id == "personal_samples":
            PERSONAL_DIR.mkdir(parents=True, exist_ok=True)
            _sync_personal_samples_state()
        elif item_id == "negative_samples":
            NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
            with AUTO_TRAIN_LOCK:
                AUTO_TRAIN_STATE["pending_negative_count"] = 0
                _save_auto_train_state_locked()
        elif item_id == "captured_audio":
            CAPTURED_DIR.mkdir(parents=True, exist_ok=True)
            _clear_auto_review_queue()
        elif item_id == "trim_history":
            TRIM_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    payload = _managed_data_payload()
    payload.update({"deleted_id": item_id, "released_bytes": previous_size})
    return payload

# --- Silero VAD (lazy-loaded) ---
_silero_vad_model = None
_silero_vad_utils = None
_SILERO_VAD_LOCK = threading.Lock()
VAD_SELECTION_PAD_START_S = 0.08
VAD_SELECTION_PAD_END_S = 0.08


def _load_silero_vad():
    """Lazy-load Silero VAD model on first use. Returns (model, utils)."""
    global _silero_vad_model, _silero_vad_utils
    if _silero_vad_model is not None:
        return _silero_vad_model, _silero_vad_utils
    with _SILERO_VAD_LOCK:
        if _silero_vad_model is not None:
            return _silero_vad_model, _silero_vad_utils
        import torch
        import silero_vad
        model = silero_vad.load_silero_vad()
        model.eval()
        _silero_vad_model = model
        _silero_vad_utils = {"torch": torch}
        return model, _silero_vad_utils


def _detect_speech_segments(wav_bytes: bytes) -> List[Dict[str, float]]:
    """Run Silero VAD on 16kHz mono WAV bytes. Return list of {start, end} in seconds."""
    model, utils = _load_silero_vad()
    torch = utils["torch"]
    import numpy as np
    from silero_vad.utils_vad import get_speech_timestamps

    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    audio_tensor = torch.from_numpy(samples)

    timestamps = get_speech_timestamps(
        audio_tensor, model, sampling_rate=16000,
        threshold=0.5, min_speech_duration_ms=150,
        min_silence_duration_ms=100, return_seconds=True,
    )
    return [{"start": round(ts["start"], 3), "end": round(ts["end"], 3)} for ts in timestamps]


def _reset_personal_samples_dir():
    _reset_audio_dir(PERSONAL_DIR)


def _reset_audio_dir(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in {".wav", ".json"}:
            try:
                p.unlink()
            except Exception:
                pass


def _list_audio_samples(directory: Path) -> List[str]:
    directory.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in directory.glob("*.wav"))


def _list_personal_samples() -> List[str]:
    return _list_audio_samples(PERSONAL_DIR)


def _list_negative_samples() -> List[str]:
    return _list_audio_samples(NEGATIVE_DIR)


def _list_captured_sample_names() -> List[str]:
    return _list_audio_samples(CAPTURED_DIR)


def _sync_trained_wake_word_artifacts() -> None:
    """One-time migration for older root-level training outputs."""
    TRAINED_WAKE_WORDS_DIR.mkdir(parents=True, exist_ok=True)
    for json_path in sorted(DATA_DIR.glob("*.json")):
        tflite_path = json_path.with_suffix(".tflite")
        if not tflite_path.exists():
            continue
        for source_path in (json_path, tflite_path):
            dest_path = TRAINED_WAKE_WORDS_DIR / source_path.name
            if not dest_path.exists() or source_path.stat().st_mtime > dest_path.stat().st_mtime:
                shutil.copy2(source_path, dest_path)
        with contextlib.suppress(Exception):
            json_path.unlink()
        with contextlib.suppress(Exception):
            tflite_path.unlink()


def _metadata_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(out):
        return None
    return out


def _metadata_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


ESPHOME_MANIFEST_SUFFIX = ".esphome.json"
ESPHOME_MANIFEST_KEYS = (
    "type",
    "wake_word",
    "author",
    "website",
    "model",
    "trained_languages",
    "version",
    "micro",
)


def _esphome_manifest(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return only fields accepted by ESPHome's micro_wake_word v2 schema."""
    return {key: metadata[key] for key in ESPHOME_MANIFEST_KEYS if key in metadata}


def _list_trained_wake_words(base_url: str = "") -> List[Dict[str, Any]]:
    _sync_trained_wake_word_artifacts()
    base = str(base_url or "").rstrip("/")
    rows: List[Dict[str, str]] = []
    seen: set[str] = set()

    for json_path in sorted(TRAINED_WAKE_WORDS_DIR.glob("*.json")):
        if json_path.name.endswith(ESPHOME_MANIFEST_SUFFIX):
            continue
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue

        model_name = str(meta.get("model") or json_path.with_suffix(".tflite").name).strip()
        model_path = TRAINED_WAKE_WORDS_DIR / Path(model_name).name
        if not model_path.exists():
            continue

        safe = json_path.stem
        if safe in seen:
            continue
        seen.add(safe)

        wake_word = str(meta.get("wake_word") or safe.replace("_", " ")).strip()
        micro = meta.get("micro") if isinstance(meta.get("micro"), dict) else {}
        native = meta.get("tater_native") if isinstance(meta.get("tater_native"), dict) else {}
        calibration = meta.get("calibration") if isinstance(meta.get("calibration"), dict) else {}
        threshold = _metadata_float(native.get("wake_threshold"))
        if threshold is None:
            threshold = _metadata_float(micro.get("probability_cutoff"))
        sliding_window = _metadata_int(native.get("wake_sliding_window"))
        if sliding_window is None:
            sliding_window = _metadata_int(micro.get("sliding_window_size"))
        close_miss_threshold = _metadata_float(native.get("close_miss_threshold"))
        recall = _metadata_float(calibration.get("recall"))
        false_accepts_per_hour = _metadata_float(calibration.get("false_accepts_per_hour"))
        json_url = f"/api/trained_wake_words/{quote(json_path.name)}"
        esphome_json_url = f"/api/trained_wake_words/{quote(safe + ESPHOME_MANIFEST_SUFFIX)}"
        model_url = f"/api/trained_wake_words/{quote(model_path.name)}"
        if base:
            json_url = f"{base}{json_url}"
            esphome_json_url = f"{base}{esphome_json_url}"
            model_url = f"{base}{model_url}"

        rows.append(
            {
                "key": safe,
                "label": wake_word or safe,
                "wake_word_name": safe,
                "wake_word": wake_word or safe,
                # `url` is retained for older trainer UIs and integrations.
                # New consumers should prefer the explicit `json_url` field.
                "url": json_url,
                "json_url": json_url,
                "esphome_json_url": esphome_json_url,
                "model_url": model_url,
                "json_file": json_path.name,
                "model_file": model_path.name,
                "threshold": round(threshold, 3) if threshold is not None else None,
                "sliding_window": sliding_window,
                "close_miss_threshold": round(close_miss_threshold, 3) if close_miss_threshold is not None else None,
                "quantization": str(meta.get("quantization") or "").strip(),
                "model_format": str(meta.get("model_format") or "").strip(),
                "sample_rate": _metadata_int(meta.get("sample_rate")),
                "calibration_recall": round(recall, 4) if recall is not None else None,
                "calibration_false_accepts_per_hour": (
                    round(false_accepts_per_hour, 6) if false_accepts_per_hour is not None else None
                ),
                "calibration_generated_at": str(calibration.get("generated_at") or "").strip(),
            }
        )
    return rows


def _request_base_url(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json_object(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)
    with contextlib.suppress(Exception):
        path.chmod(0o600)


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _config_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    if token in {"1", "true", "yes", "on", "enabled"}:
        return True
    if token in {"0", "false", "no", "off", "disabled"}:
        return False
    return bool(default)


def _normalize_http_base_url(value: Any, *, allow_empty: bool = True) -> str:
    token = str(value or "").strip().rstrip("/")
    if not token and allow_empty:
        return ""
    if not token.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return token


def _normalize_stt_engine(value: Any) -> str:
    token = str(value or DEFAULT_STT_ENGINE).strip().lower().replace("-", "_")
    aliases = {
        "faster": STT_ENGINE_FASTER_WHISPER,
        "fasterwhisper": STT_ENGINE_FASTER_WHISPER,
        "parakeet": STT_ENGINE_PARAKEET_ONNX,
        "onnx_parakeet": STT_ENGINE_PARAKEET_ONNX,
        "mlx": STT_ENGINE_MLX_WHISPER,
        "mlxwhisper": STT_ENGINE_MLX_WHISPER,
    }
    token = aliases.get(token, token)
    if token not in SUPPORTED_STT_ENGINES:
        raise ValueError("STT engine must be Faster Whisper, Parakeet ONNX, or MLX Whisper.")
    return token


def _managed_stt_model(engine: Any, language: Any = DEFAULT_LANGUAGE) -> str:
    token = _normalize_stt_engine(engine)
    language_token = str(language or DEFAULT_LANGUAGE).strip().lower().replace("-", "_")
    english = language_token == "en" or language_token.startswith("en_")
    if token == STT_ENGINE_FASTER_WHISPER:
        return (
            DEFAULT_FASTER_WHISPER_EN_MODEL
            if english
            else DEFAULT_FASTER_WHISPER_MULTILINGUAL_MODEL
        )
    if token == STT_ENGINE_MLX_WHISPER:
        return (
            DEFAULT_MLX_WHISPER_EN_MODEL
            if english
            else DEFAULT_MLX_WHISPER_MULTILINGUAL_MODEL
        )
    return DEFAULT_PARAKEET_ONNX_MODEL


def _stt_engine_catalog(language: Any = DEFAULT_LANGUAGE) -> List[Dict[str, Any]]:
    return [
        {
            "value": STT_ENGINE_FASTER_WHISPER,
            "label": "Faster Whisper",
            "model": _managed_stt_model(STT_ENGINE_FASTER_WHISPER, language),
            "recommended": True,
        },
        {
            "value": STT_ENGINE_PARAKEET_ONNX,
            "label": "Parakeet ONNX",
            "model": _managed_stt_model(STT_ENGINE_PARAKEET_ONNX, language),
        },
        {
            "value": STT_ENGINE_MLX_WHISPER,
            "label": "MLX Whisper",
            "model": _managed_stt_model(STT_ENGINE_MLX_WHISPER, language),
        },
    ]


def _normalize_auto_train_config(values: Dict[str, Any] | None, *, base: Dict[str, Any] | None = None) -> Dict[str, Any]:
    incoming = values if isinstance(values, dict) else {}
    source = {**AUTO_TRAIN_DEFAULT_CONFIG, **(base or {}), **incoming}
    schedule_hours = _bounded_int(source.get("schedule_hours"), 24, 0, 24 * 30)
    language = str(source.get("language") or DEFAULT_LANGUAGE).strip().lower().replace("-", "_")
    language = re.sub(r"[^a-z0-9_]", "", language) or DEFAULT_LANGUAGE
    return {
        "enabled": _config_bool(source.get("enabled")),
        "wake_phrase": str(source.get("wake_phrase") or "").strip(),
        "language": language,
        "english_accent": normalize_english_accent(
            source.get("english_accent"), language
        ),
        "stt_engine": _normalize_stt_engine(source.get("stt_engine")),
        "minimum_transcript_chars": _bounded_int(source.get("minimum_transcript_chars"), 2, 1, 100),
        "delete_confirmed_wakes": _config_bool(source.get("delete_confirmed_wakes")),
        "promote_close_misses": _config_bool(source.get("promote_close_misses")),
        "schedule_hours": schedule_hours,
        "minimum_new_negatives": _bounded_int(source.get("minimum_new_negatives"), 3, 1, 10000),
        "advertised_base_url": _normalize_http_base_url(source.get("advertised_base_url")),
        "tater_url": _normalize_http_base_url(source.get("tater_url"), allow_empty=False),
        "tater_link_token": str(source.get("tater_link_token") or "").strip(),
        "tater_link_id": str(source.get("tater_link_id") or "").strip(),
        "tater_linked_at": str(source.get("tater_linked_at") or "").strip(),
        "tater_link_tater_name": str(source.get("tater_link_tater_name") or "").strip(),
        "notify_satellites": _config_bool(source.get("notify_satellites"), True),
    }


try:
    AUTO_TRAIN_CONFIG: Dict[str, Any] = _normalize_auto_train_config(
        _read_json_object(AUTO_TRAIN_CONFIG_FILE)
    )
except ValueError:
    AUTO_TRAIN_CONFIG = dict(AUTO_TRAIN_DEFAULT_CONFIG)
AUTO_TRAIN_STATE: Dict[str, Any] = {
    **AUTO_TRAIN_DEFAULT_STATE,
    **_read_json_object(AUTO_TRAIN_STATE_FILE),
}


def _save_auto_train_config_locked() -> None:
    _write_json_object(AUTO_TRAIN_CONFIG_FILE, AUTO_TRAIN_CONFIG)


def _save_auto_train_state_locked() -> None:
    persisted = {key: AUTO_TRAIN_STATE.get(key) for key in AUTO_TRAIN_DEFAULT_STATE}
    _write_json_object(AUTO_TRAIN_STATE_FILE, persisted)


def _parse_iso_datetime(value: Any) -> datetime | None:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _schedule_next_auto_run_locked(*, from_time: datetime | None = None) -> None:
    hours = int(AUTO_TRAIN_CONFIG.get("schedule_hours") or 0)
    if hours <= 0 or not AUTO_TRAIN_CONFIG.get("enabled"):
        AUTO_TRAIN_STATE["next_run_at"] = ""
    else:
        base = from_time or _utc_now()
        AUTO_TRAIN_STATE["next_run_at"] = (base + timedelta(hours=hours)).isoformat()
    _save_auto_train_state_locked()


def _public_auto_train_config() -> Dict[str, Any]:
    with AUTO_TRAIN_LOCK:
        config = {key: value for key, value in AUTO_TRAIN_CONFIG.items() if key != "tater_link_token"}
        config["tater_linked"] = bool(
            AUTO_TRAIN_CONFIG.get("tater_link_token")
            and AUTO_TRAIN_CONFIG.get("tater_link_id")
        )
        return config


def _auto_train_status_payload() -> Dict[str, Any]:
    with AUTO_TRAIN_LOCK:
        language = AUTO_TRAIN_CONFIG.get("language") or DEFAULT_LANGUAGE
        return {
            "config": _public_auto_train_config(),
            "state": dict(AUTO_TRAIN_STATE),
            "runtime": dict(AUTO_TRAIN_RUNTIME),
            "stt_engines": _stt_engine_catalog(language),
            "advertised_base_url": _advertised_base_url(),
            "trainer_link": _tater_link_public_status(),
        }


def _discover_lan_ipv4() -> str:
    override = str(os.environ.get("REC_ADVERTISED_HOST") or "").strip()
    if override and override not in {"0.0.0.0", "127.0.0.1", "localhost", "::1"}:
        return override

    now = time.time()
    cached_value = str(LAN_ADDRESS_CACHE.get("value") or "")
    if cached_value and (now - float(LAN_ADDRESS_CACHE.get("fetched_at") or 0.0)) < 30:
        return cached_value

    candidates: List[str] = []
    with contextlib.suppress(Exception):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("192.0.2.1", 9))
            candidates.append(str(sock.getsockname()[0]))
        finally:
            sock.close()
    with contextlib.suppress(Exception):
        for row in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_DGRAM):
            candidates.append(str(row[4][0]))

    with contextlib.suppress(Exception):
        proc = subprocess.run(
            ["/sbin/ifconfig"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        blocks = re.split(r"(?m)(?=^[^\s].*?: flags=)", proc.stdout or "")
        interface_rows: List[tuple[int, str]] = []
        for block in blocks:
            name_match = re.match(r"^([^:]+):", block)
            address_match = re.search(r"(?m)^\s+inet\s+(\d+(?:\.\d+){3})\b", block)
            if not name_match or not address_match or "status: active" not in block:
                continue
            name = name_match.group(1)
            if name == "lo0" or name.startswith(("utun", "awdl", "llw", "ap")):
                continue
            priority = 0 if name == "en0" else 1 if name == "en1" else 10
            interface_rows.append((priority, address_match.group(1)))
        candidates.extend(address for _priority, address in sorted(interface_rows))

    for candidate in candidates:
        if candidate and not candidate.startswith("127.") and candidate != "0.0.0.0":
            LAN_ADDRESS_CACHE["value"] = candidate
            LAN_ADDRESS_CACHE["fetched_at"] = now
            return candidate
    LAN_ADDRESS_CACHE["fetched_at"] = now
    return ""


def _advertised_base_url(request: Request | None = None) -> str:
    env_url = str(os.environ.get("REC_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    with AUTO_TRAIN_LOCK:
        configured_url = str(AUTO_TRAIN_CONFIG.get("advertised_base_url") or "").strip().rstrip("/")
    if configured_url:
        return configured_url
    if env_url:
        return env_url

    request_url = _request_base_url(request) if request is not None else ""
    request_host = str(request.url.hostname or "").lower() if request is not None else ""
    if request_url and request_host not in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}:
        return request_url

    host = _discover_lan_ipv4()
    if not host:
        return request_url
    scheme = str(request.url.scheme or "http") if request is not None else "http"
    port = request.url.port if request is not None else None
    if port is None:
        port = _bounded_int(os.environ.get("REC_PORT"), 8789, 1, 65535)
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    return f"{scheme}://{host}{'' if default_port else f':{port}'}"


def _tater_link_public_status() -> Dict[str, Any]:
    with AUTO_TRAIN_LOCK:
        return {
            "linked": bool(
                AUTO_TRAIN_CONFIG.get("tater_link_token")
                and AUTO_TRAIN_CONFIG.get("tater_link_id")
            ),
            "trainer_id": str(AUTO_TRAIN_CONFIG.get("tater_link_id") or "").strip(),
            "linked_at": str(AUTO_TRAIN_CONFIG.get("tater_linked_at") or "").strip(),
            "tater_name": str(AUTO_TRAIN_CONFIG.get("tater_link_tater_name") or "").strip(),
        }


def _claim_tater_link(tater_url: Any, pairing_code: Any) -> Dict[str, Any]:
    base_url = _normalize_http_base_url(tater_url, allow_empty=False)
    code = "".join(ch for ch in str(pairing_code or "").upper() if ch.isalnum())
    if len(code) != 8:
        raise ValueError("Enter the complete pairing code shown by Tater.")
    publish_base_url = _normalize_http_base_url(_advertised_base_url(), allow_empty=False)
    with AUTO_TRAIN_LOCK:
        trainer_id = str(AUTO_TRAIN_CONFIG.get("tater_link_id") or "").strip() or secrets.token_hex(12)

    request = URLRequest(
        f"{base_url}/api/tater/satellite/v1/trainer/link/claim",
        data=json.dumps(
            {
                "pairing_code": code,
                "trainer_id": trainer_id,
                "trainer_name": "Wake Word Trainer",
                "trainer_url": publish_base_url,
                "publish_base_url": publish_base_url,
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "microWakeWord-Trainer/tater-link",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read(64 * 1024).decode("utf-8"))
    except HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            error_payload = json.loads(exc.read(64 * 1024).decode("utf-8"))
            if isinstance(error_payload, dict):
                detail = str(error_payload.get("detail") or error_payload.get("error") or "").strip()
        raise ValueError(detail or f"Tater rejected the pairing code (HTTP {exc.code}).") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not reach Tater: {exc}") from exc

    if not isinstance(payload, dict) or not bool(payload.get("ok")):
        raise ValueError(str((payload or {}).get("error") or "Tater pairing failed."))
    link_token = str(payload.get("token") or "").strip()
    if len(link_token) < 32:
        raise ValueError("Tater pairing response did not contain valid link credentials.")

    linked_at = str(payload.get("linked_at") or _iso_now()).strip()
    tater_name = str(payload.get("tater_name") or "Tater").strip() or "Tater"
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_CONFIG["tater_url"] = base_url
        AUTO_TRAIN_CONFIG["tater_link_token"] = link_token
        AUTO_TRAIN_CONFIG["tater_link_id"] = trainer_id
        AUTO_TRAIN_CONFIG["tater_linked_at"] = linked_at
        AUTO_TRAIN_CONFIG["tater_link_tater_name"] = tater_name
        _save_auto_train_config_locked()
    return {
        "ok": True,
        "message": "Tater linked successfully.",
        **_tater_link_public_status(),
    }


def _unlink_tater() -> Dict[str, Any]:
    with AUTO_TRAIN_LOCK:
        base_url = str(AUTO_TRAIN_CONFIG.get("tater_url") or "").strip().rstrip("/")
        link_token = str(AUTO_TRAIN_CONFIG.get("tater_link_token") or "").strip()
    remote_error = ""
    if base_url and link_token:
        request = URLRequest(
            f"{base_url}/api/tater/satellite/v1/trainer/link/unlink",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "X-Tater-Trainer-Token": link_token,
                "User-Agent": "microWakeWord-Trainer/tater-link",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=10):
                pass
        except Exception as exc:
            remote_error = str(exc)
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_CONFIG["tater_link_token"] = ""
        AUTO_TRAIN_CONFIG["tater_link_id"] = ""
        AUTO_TRAIN_CONFIG["tater_linked_at"] = ""
        AUTO_TRAIN_CONFIG["tater_link_tater_name"] = ""
        _save_auto_train_config_locked()
    return {
        "ok": True,
        "message": "Tater link removed." if not remote_error else "Local Tater link removed; Tater could not be reached.",
        "remote_error": remote_error,
        **_tater_link_public_status(),
    }


def _normalize_transcript_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("_", " ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _transcript_contains_wake_phrase(transcript: Any, wake_phrase: Any) -> bool:
    normalized_transcript = _normalize_transcript_text(transcript)
    normalized_phrase = _normalize_transcript_text(wake_phrase)
    if not normalized_transcript or not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_transcript} "


def _wake_phrase_similarity(transcript: Any, wake_phrase: Any) -> float:
    transcript_words = _normalize_transcript_text(transcript).split()
    phrase_words = _normalize_transcript_text(wake_phrase).split()
    if not transcript_words or not phrase_words:
        return 0.0
    if _transcript_contains_wake_phrase(transcript, wake_phrase):
        return 1.0

    phrase_token = "".join(phrase_words)
    minimum_words = max(1, len(phrase_words) - 1)
    maximum_words = min(len(transcript_words), len(phrase_words) + 1)
    best_score = 0.0
    for word_count in range(minimum_words, maximum_words + 1):
        for start in range(0, len(transcript_words) - word_count + 1):
            candidate = "".join(transcript_words[start : start + word_count])
            best_score = max(
                best_score,
                SequenceMatcher(None, candidate, phrase_token).ratio(),
            )
    return best_score


def _captured_event_is_close_miss(metadata: Dict[str, Any]) -> bool:
    event_type = str(metadata.get("event_type") or "captured").strip().lower()
    return "close" in event_type


def _captured_event_is_auto_reviewable(
    metadata: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> bool:
    if _parse_bool(metadata.get("blocked_by_vad")):
        return False
    event_type = str(metadata.get("event_type") or "captured").strip().lower()
    if "close" in event_type:
        return bool((config or {}).get("promote_close_misses"))
    return event_type in {"captured", "trigger", "false_trigger"} or "wake" in event_type or "detect" in event_type


@contextlib.contextmanager
def _auto_train_model_environment():
    AUTO_TRAIN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    keys = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["HF_HOME"] = str(AUTO_TRAIN_MODEL_DIR)
    os.environ["HF_HUB_CACHE"] = str(AUTO_TRAIN_MODEL_DIR / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(AUTO_TRAIN_MODEL_DIR / "hub")
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _record_stt_runtime(
    *,
    engine: str,
    model: str,
    device: str,
    compute_type: str,
) -> None:
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_STATE["last_stt_engine"] = engine
        AUTO_TRAIN_STATE["last_stt_model"] = model
        AUTO_TRAIN_STATE["last_stt_device"] = device
        AUTO_TRAIN_STATE["last_stt_compute_type"] = compute_type
        _save_auto_train_state_locked()


def _resolve_faster_whisper_runtime() -> tuple[str, str]:
    cuda_devices = 0
    with contextlib.suppress(Exception):
        import ctranslate2

        cuda_devices = int(ctranslate2.get_cuda_device_count())
    return ("cuda", "float16") if cuda_devices > 0 else ("cpu", "int8")


def _load_faster_whisper_model(*, model_name: str, device: str, compute_type: str):
    cache_key = (model_name, device, compute_type)
    with FASTER_WHISPER_MODEL_LOCK:
        cached = FASTER_WHISPER_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(f"faster-whisper is unavailable: {exc}") from exc

        AUTO_TRAIN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(AUTO_TRAIN_MODEL_DIR),
        )
        FASTER_WHISPER_MODEL_CACHE.clear()
        FASTER_WHISPER_MODEL_CACHE[cache_key] = model
        return model


def _transcribe_capture_with_faster_whisper(
    audio_path: Path,
    *,
    model: str,
    language: str,
) -> str:
    device, compute_type = _resolve_faster_whisper_runtime()
    whisper_model = _load_faster_whisper_model(
        model_name=model,
        device=device,
        compute_type=compute_type,
    )
    with FASTER_WHISPER_TRANSCRIBE_LOCK:
        segments, _info = whisper_model.transcribe(
            str(audio_path),
            language=language or None,
            beam_size=1,
            condition_on_previous_text=False,
        )
        transcript = re.sub(
            r"\s+",
            " ",
            " ".join(str(segment.text or "").strip() for segment in segments),
        ).strip()
    _record_stt_runtime(
        engine=STT_ENGINE_FASTER_WHISPER,
        model=model,
        device=device,
        compute_type=compute_type,
    )
    return transcript


def _transcribe_capture_with_faster_whisper_guided(
    audio_path: Path,
    *,
    model: str,
    language: str,
    wake_phrase: str,
) -> str:
    normalized_phrase = _normalize_transcript_text(wake_phrase)
    if not normalized_phrase:
        return ""
    device, compute_type = _resolve_faster_whisper_runtime()
    whisper_model = _load_faster_whisper_model(
        model_name=model,
        device=device,
        compute_type=compute_type,
    )
    with FASTER_WHISPER_TRANSCRIBE_LOCK:
        segments, _info = whisper_model.transcribe(
            str(audio_path),
            language=language or None,
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=f'The wake phrase is "{normalized_phrase}".',
            hotwords=normalized_phrase,
        )
        return re.sub(
            r"\s+",
            " ",
            " ".join(str(segment.text or "").strip() for segment in segments),
        ).strip()


def _transcribe_capture_with_mlx(audio_path: Path, *, model: str, language: str) -> str:
    try:
        import mlx_whisper
    except Exception as exc:
        raise RuntimeError(f"mlx-whisper is unavailable: {exc}") from exc

    kwargs: Dict[str, Any] = {
        "path_or_hf_repo": model,
        "verbose": False,
    }
    if language:
        kwargs["language"] = language
    with _auto_train_model_environment():
        with MLX_WHISPER_TRANSCRIBE_LOCK:
            try:
                result = mlx_whisper.transcribe(str(audio_path), **kwargs)
            except TypeError:
                result = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=model)
    text = result.get("text") if isinstance(result, dict) else result
    transcript = re.sub(r"\s+", " ", str(text or "")).strip()
    _record_stt_runtime(
        engine=STT_ENGINE_MLX_WHISPER,
        model=model,
        device="mps",
        compute_type="mlx",
    )
    return transcript


def _parakeet_onnx_providers() -> List[str]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(f"onnxruntime is unavailable: {exc}") from exc
    available = [str(value) for value in ort.get_available_providers()]
    preferred = [
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    resolved = [provider for provider in preferred if provider in set(available)]
    if not resolved:
        raise RuntimeError("ONNX Runtime has no usable Core ML or CPU execution provider.")
    return resolved


def _load_parakeet_onnx_model():
    try:
        import onnx_asr
    except Exception as exc:
        raise RuntimeError(f"onnx-asr is unavailable: {exc}") from exc

    providers = tuple(_parakeet_onnx_providers())
    cache_key = (
        DEFAULT_PARAKEET_ONNX_MODEL,
        DEFAULT_PARAKEET_ONNX_QUANTIZATION,
        providers,
    )
    with PARAKEET_ONNX_MODEL_LOCK:
        cached = PARAKEET_ONNX_MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        suffix = (
            f".{DEFAULT_PARAKEET_ONNX_QUANTIZATION}"
            if DEFAULT_PARAKEET_ONNX_QUANTIZATION
            else ""
        )
        model_patterns = [
            "config.json",
            "vocab.txt",
            f"encoder-model{suffix}.onnx",
            f"encoder-model{suffix}.onnx.data",
            f"decoder_joint-model{suffix}.onnx",
            f"decoder_joint-model{suffix}.onnx.data",
        ]
        required_model_files = [
            "config.json",
            "vocab.txt",
            f"encoder-model{suffix}.onnx",
            f"decoder_joint-model{suffix}.onnx",
        ]
        if not DEFAULT_PARAKEET_ONNX_QUANTIZATION:
            required_model_files.append("encoder-model.onnx.data")
        with _auto_train_model_environment():
            snapshot_root = AUTO_TRAIN_MODEL_DIR
            if not all(
                (AUTO_TRAIN_MODEL_DIR / filename).is_file()
                for filename in required_model_files
            ):
                from huggingface_hub import snapshot_download

                snapshot_root = Path(
                    snapshot_download(
                        repo_id=DEFAULT_PARAKEET_ONNX_REPO,
                        local_dir=str(AUTO_TRAIN_MODEL_DIR),
                        allow_patterns=model_patterns,
                    )
                )
            model = onnx_asr.load_model(
                DEFAULT_PARAKEET_ONNX_MODEL,
                str(snapshot_root),
                quantization=DEFAULT_PARAKEET_ONNX_QUANTIZATION,
                providers=list(providers),
            )
        PARAKEET_ONNX_MODEL_CACHE.clear()
        PARAKEET_ONNX_MODEL_CACHE[cache_key] = model
        return model


def _normalized_wav_float32(audio_path: Path):
    import numpy as np

    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2 or sample_rate != TARGET_SAMPLE_RATE or channels < 1:
        raise RuntimeError("STT input must be 16 kHz, 16-bit PCM WAV audio.")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape((-1, channels)).mean(axis=1)
    return samples / 32768.0


def _transcribe_capture_with_parakeet(audio_path: Path, *, model: str, language: str) -> str:
    parakeet_model = _load_parakeet_onnx_model()
    kwargs: Dict[str, Any] = {
        "sample_rate": TARGET_SAMPLE_RATE,
        "channel": "mean",
    }
    if language:
        kwargs["language"] = language
    with PARAKEET_ONNX_TRANSCRIBE_LOCK:
        result = parakeet_model.recognize(
            _normalized_wav_float32(audio_path),
            **kwargs,
        )
    providers = _parakeet_onnx_providers()
    _record_stt_runtime(
        engine=STT_ENGINE_PARAKEET_ONNX,
        model=model,
        device=providers[0],
        compute_type=DEFAULT_PARAKEET_ONNX_QUANTIZATION,
    )
    return re.sub(r"\s+", " ", str(result or "")).strip()


def _transcribe_capture(audio_path: Path, *, engine: str, language: str) -> str:
    token = _normalize_stt_engine(engine)
    model = _managed_stt_model(token, language)
    if token == STT_ENGINE_FASTER_WHISPER:
        return _transcribe_capture_with_faster_whisper(
            audio_path,
            model=model,
            language=language,
        )
    if token == STT_ENGINE_PARAKEET_ONNX:
        return _transcribe_capture_with_parakeet(
            audio_path,
            model=model,
            language=language,
        )
    return _transcribe_capture_with_mlx(
        audio_path,
        model=model,
        language=language,
    )


def _clear_stt_model_caches(*, keep_engine: str) -> None:
    token = _normalize_stt_engine(keep_engine)
    cleared = False
    if token != STT_ENGINE_FASTER_WHISPER:
        with FASTER_WHISPER_TRANSCRIBE_LOCK:
            with FASTER_WHISPER_MODEL_LOCK:
                cleared = bool(FASTER_WHISPER_MODEL_CACHE) or cleared
                FASTER_WHISPER_MODEL_CACHE.clear()
    if token != STT_ENGINE_PARAKEET_ONNX:
        with PARAKEET_ONNX_TRANSCRIBE_LOCK:
            with PARAKEET_ONNX_MODEL_LOCK:
                cleared = bool(PARAKEET_ONNX_MODEL_CACHE) or cleared
                PARAKEET_ONNX_MODEL_CACHE.clear()
    if token != STT_ENGINE_MLX_WHISPER:
        with contextlib.suppress(Exception):
            import mlx.core as mlx_core

            mlx_core.clear_cache()
    if cleared:
        gc.collect()


def _queue_auto_review(file_name: str) -> bool:
    safe_file_name = Path(str(file_name or "")).name
    if not safe_file_name:
        return False
    with AUTO_TRAIN_LOCK:
        if safe_file_name in AUTO_TRAIN_QUEUED_FILES:
            return False
        AUTO_TRAIN_QUEUED_FILES.add(safe_file_name)
    AUTO_TRAIN_REVIEW_QUEUE.put(safe_file_name)
    AUTO_TRAIN_WAKE_EVENT.set()
    return True


def _queue_pending_auto_reviews(*, force: bool = False) -> int:
    queued = 0
    with AUTO_TRAIN_LOCK:
        config = dict(AUTO_TRAIN_CONFIG)
    if not config.get("enabled"):
        return queued
    CAPTURED_DIR.mkdir(parents=True, exist_ok=True)
    for audio_path in sorted(CAPTURED_DIR.glob("*.wav")):
        metadata = _load_sidecar_json(audio_path)
        if not _captured_event_is_auto_reviewable(metadata, config):
            continue
        status = str(metadata.get("auto_review_status") or "").strip()
        if status == "transcribing":
            metadata.pop("auto_review_status", None)
            _write_sidecar_json(audio_path, metadata)
            status = ""
        if force and status in {"error", "no_speech", "wake_phrase_ambiguous"}:
            metadata.pop("auto_review_status", None)
            _write_sidecar_json(audio_path, metadata)
            status = ""
        if (
            status == "wake_phrase_detected"
            and config.get("delete_confirmed_wakes")
            and not _captured_event_is_close_miss(metadata)
        ):
            status = ""
        if status:
            continue
        if _queue_auto_review(audio_path.name):
            queued += 1
    return queued


def _record_auto_review_result(*, file_name: str, transcript: str = "", result: str = "", error: str = "") -> None:
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_STATE["last_review_at"] = _iso_now()
        AUTO_TRAIN_STATE["last_review_file"] = file_name
        AUTO_TRAIN_STATE["last_review_transcript"] = transcript
        AUTO_TRAIN_STATE["last_review_result"] = result
        AUTO_TRAIN_STATE["last_review_error"] = error
        _save_auto_train_state_locked()


def _auto_review_capture(file_name: str) -> None:
    try:
        with AUTO_TRAIN_LOCK:
            config = dict(AUTO_TRAIN_CONFIG)
            AUTO_TRAIN_RUNTIME["review_running"] = True
            AUTO_TRAIN_RUNTIME["review_file"] = file_name
        if not config.get("enabled"):
            return
        wake_phrase = str(config.get("wake_phrase") or "").strip()
        if not wake_phrase:
            _record_auto_review_result(file_name=file_name, result="waiting_for_wake_phrase")
            return

        try:
            audio_path = _resolve_audio_path(CAPTURED_DIR, file_name)
        except FileNotFoundError:
            return
        metadata = _load_sidecar_json(audio_path)
        is_close_miss = _captured_event_is_close_miss(metadata)
        status = str(metadata.get("auto_review_status") or "").strip()
        if (
            status == "wake_phrase_detected"
            and config.get("delete_confirmed_wakes")
            and not is_close_miss
        ):
            transcript = str(metadata.get("transcript") or "")
            _remove_audio_with_sidecar(audio_path)
            _record_auto_review_result(
                file_name=file_name,
                transcript=transcript,
                result="deleted_confirmed_wake",
            )
            return
        if status or not _captured_event_is_auto_reviewable(metadata, config):
            return
        captured_wake_phrase = str(metadata.get("wake_word") or "").strip()
        if captured_wake_phrase and _normalize_transcript_text(captured_wake_phrase) != _normalize_transcript_text(wake_phrase):
            metadata["auto_review_status"] = "different_wake_phrase"
            metadata["auto_review_reason"] = (
                f"Capture is for '{captured_wake_phrase}', not configured phrase '{wake_phrase}'; left for manual review."
            )
            metadata["auto_reviewed_at"] = _iso_now()
            _write_sidecar_json(audio_path, metadata)
            _record_auto_review_result(file_name=file_name, result="different_wake_phrase")
            return

        metadata["auto_review_status"] = "transcribing"
        metadata["auto_reviewed_at"] = _iso_now()
        metadata["auto_review_wake_phrase"] = wake_phrase
        stt_engine = _normalize_stt_engine(config.get("stt_engine"))
        metadata["auto_review_stt_engine"] = stt_engine
        metadata["auto_review_stt_model"] = _managed_stt_model(
            stt_engine,
            config.get("language"),
        )
        _write_sidecar_json(audio_path, metadata)

        transcript = _transcribe_capture(
            audio_path,
            engine=stt_engine,
            language=str(config.get("language") or DEFAULT_LANGUAGE),
        )
        normalized = _normalize_transcript_text(transcript)
        metadata = _load_sidecar_json(audio_path)
        metadata["transcript"] = transcript
        metadata["transcribed_at"] = _iso_now()

        if len(normalized) < int(config.get("minimum_transcript_chars") or 2):
            metadata["auto_review_status"] = "no_speech"
            metadata["auto_review_reason"] = "STT did not return enough text; left for manual review."
            _write_sidecar_json(audio_path, metadata)
            _record_auto_review_result(file_name=file_name, transcript=transcript, result="no_speech")
            return

        phrase_similarity = _wake_phrase_similarity(transcript, wake_phrase)
        phrase_detected = _transcript_contains_wake_phrase(transcript, wake_phrase)
        match_method = "exact" if phrase_detected else ""
        metadata["auto_review_phrase_similarity"] = round(phrase_similarity, 4)

        if (
            not phrase_detected
            and phrase_similarity >= WAKE_PHRASE_GUIDANCE_MIN_SIMILARITY
            and stt_engine == STT_ENGINE_FASTER_WHISPER
        ):
            guided_transcript = _transcribe_capture_with_faster_whisper_guided(
                audio_path,
                model=str(metadata["auto_review_stt_model"]),
                language=str(config.get("language") or DEFAULT_LANGUAGE),
                wake_phrase=wake_phrase,
            )
            metadata["auto_review_guided_transcript"] = guided_transcript
            if _transcript_contains_wake_phrase(guided_transcript, wake_phrase):
                phrase_detected = True
                match_method = "guided_close_match"

        if match_method:
            metadata["auto_review_match_method"] = match_method

        if phrase_detected:
            guided_confirmation = match_method == "guided_close_match"
            if is_close_miss:
                metadata["auto_review_status"] = "approved_positive"
                metadata["auto_review_reason"] = (
                    "Close miss was confirmed as the configured wake phrase and promoted to a positive sample."
                    if guided_confirmation
                    else "Close miss contained the configured wake phrase and was promoted to a positive sample."
                )
                metadata["auto_positive"] = True
                _write_sidecar_json(audio_path, metadata)
                _move_captured_audio(
                    file_name,
                    PERSONAL_DIR,
                    target_prefix="sample",
                    review_status="auto_approved_personal",
                )
                _record_auto_review_result(
                    file_name=file_name,
                    transcript=transcript,
                    result="promoted_close_miss",
                )
                return
            if config.get("delete_confirmed_wakes"):
                _remove_audio_with_sidecar(audio_path)
                _record_auto_review_result(
                    file_name=file_name,
                    transcript=transcript,
                    result="deleted_confirmed_wake",
                )
                return
            metadata["auto_review_status"] = "wake_phrase_detected"
            metadata["auto_review_reason"] = (
                "Wake phrase confirmed by a guided second STT pass; left for manual positive review."
                if guided_confirmation
                else "Wake phrase found in transcript; left for manual positive review."
            )
            _write_sidecar_json(audio_path, metadata)
            _record_auto_review_result(file_name=file_name, transcript=transcript, result="wake_phrase_detected")
            return

        if phrase_similarity >= WAKE_PHRASE_GUIDANCE_MIN_SIMILARITY:
            metadata["auto_review_status"] = "wake_phrase_ambiguous"
            metadata["auto_review_reason"] = (
                "STT sounded close to the configured wake phrase but could not confirm it; "
                "left for manual review."
            )
            _write_sidecar_json(audio_path, metadata)
            _record_auto_review_result(
                file_name=file_name,
                transcript=transcript,
                result="wake_phrase_ambiguous",
            )
            return

        if is_close_miss:
            metadata["auto_review_status"] = "close_miss_phrase_not_detected"
            metadata["auto_review_reason"] = (
                "Close miss did not contain the configured wake phrase; left for manual review."
            )
            _write_sidecar_json(audio_path, metadata)
            _record_auto_review_result(
                file_name=file_name,
                transcript=transcript,
                result="close_miss_phrase_not_detected",
            )
            return

        metadata["auto_review_status"] = "approved_negative"
        metadata["auto_review_reason"] = "Wake phrase was not found in the STT transcript."
        metadata["auto_negative"] = True
        _write_sidecar_json(audio_path, metadata)
        _move_captured_audio(
            file_name,
            NEGATIVE_DIR,
            target_prefix="negative",
            review_status="auto_approved_negative",
        )
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["pending_negative_count"] = int(AUTO_TRAIN_STATE.get("pending_negative_count") or 0) + 1
            _save_auto_train_state_locked()
        _record_auto_review_result(file_name=file_name, transcript=transcript, result="approved_negative")
    except Exception as exc:
        error = str(exc)
        with contextlib.suppress(Exception):
            audio_path = _resolve_audio_path(CAPTURED_DIR, file_name)
            metadata = _load_sidecar_json(audio_path)
            metadata["auto_review_status"] = "error"
            metadata["auto_review_error"] = error
            metadata["auto_reviewed_at"] = _iso_now()
            _write_sidecar_json(audio_path, metadata)
        _record_auto_review_result(file_name=file_name, result="error", error=error)
    finally:
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_RUNTIME["review_running"] = False
            AUTO_TRAIN_RUNTIME["review_file"] = ""


def _notify_tater_satellites(wake_word_name: str = "") -> Dict[str, Any]:
    with AUTO_TRAIN_LOCK:
        config = dict(AUTO_TRAIN_CONFIG)
    if not config.get("notify_satellites"):
        return {"ok": True, "skipped": True, "message": "Satellite notification is disabled."}

    base_url = str(config.get("tater_url") or "").rstrip("/")
    settings_endpoint = f"{base_url}/api/tater/satellite/v1/trainer/wake-word"
    headers = {"Content-Type": "application/json", "User-Agent": "microWakeWord-Trainer/auto-train"}
    token = str(config.get("tater_link_token") or "").strip()
    if not token:
        return {
            "ok": False,
            "error": "Wake Word Trainer is not linked to Tater. Use Link Tater first.",
        }
    headers["X-Tater-Trainer-Token"] = token

    try:
        target_key = safe_name(wake_word_name or config.get("wake_phrase") or "")
        public_base_url = _advertised_base_url()
        wake_words = _list_trained_wake_words(public_base_url)
        target = next(
            (row for row in wake_words if str(row.get("key") or "").strip() == target_key),
            None,
        )
        if not isinstance(target, dict):
            raise FileNotFoundError(f"Trained wake word is not available: {target_key}")
        wake_word_url = str(target.get("json_url") or "").strip()
        if not wake_word_url.startswith(("http://", "https://")):
            raise ValueError("The trained wake-word JSON needs an advertised http(s) URL.")

        body = json.dumps(
            {
                "wake_word_name": target_key,
                "wake_word_url": wake_word_url,
            }
        ).encode("utf-8")
        request = URLRequest(settings_endpoint, data=body, headers=headers, method="POST")
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        push = payload.get("push") if isinstance(payload, dict) and isinstance(payload.get("push"), dict) else {}
        pushed_count = push.get("count")
        count = max(0, int(pushed_count)) if isinstance(pushed_count, (int, float)) else 0

        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["last_notify_at"] = _iso_now()
            AUTO_TRAIN_STATE["last_notify_count"] = count
            AUTO_TRAIN_STATE["last_notify_error"] = ""
            _save_auto_train_state_locked()
        return {
            "ok": True,
            "count": count,
            "wake_word": str(target.get("wake_word") or target_key),
            "wake_word_name": target_key,
            "wake_word_url": wake_word_url,
        }
    except HTTPError as exc:
        detail = ""
        with contextlib.suppress(Exception):
            error_payload = json.loads(exc.read().decode("utf-8"))
            if isinstance(error_payload, dict):
                detail = str(error_payload.get("detail") or error_payload.get("error") or "").strip()
        error = detail or f"Tater rejected the wake word (HTTP {exc.code})."
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["last_notify_at"] = _iso_now()
            AUTO_TRAIN_STATE["last_notify_count"] = None
            AUTO_TRAIN_STATE["last_notify_error"] = error
            _save_auto_train_state_locked()
        return {"ok": False, "error": error}
    except Exception as exc:
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["last_notify_at"] = _iso_now()
            AUTO_TRAIN_STATE["last_notify_count"] = None
            AUTO_TRAIN_STATE["last_notify_error"] = str(exc)
            _save_auto_train_state_locked()
        return {"ok": False, "error": str(exc)}


def _try_acquire_training_run_lock():
    TRAINING_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_file = TRAINING_LOCK_FILE.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None

    try:
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": _iso_now(),
                }
            )
            + "\n"
        )
        lock_file.flush()
    except Exception:
        _release_training_run_lock(lock_file)
        raise
    return lock_file


def _release_training_run_lock(lock_file) -> None:
    if lock_file is None:
        return
    with contextlib.suppress(Exception):
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    with contextlib.suppress(Exception):
        lock_file.close()


def _terminate_training_process_tree(
    proc: subprocess.Popen,
    *,
    graceful_timeout: float = 12.0,
    kill_timeout: float = 3.0,
) -> bool:
    if proc.poll() is not None:
        return True

    process_group = None
    if os.name == "posix":
        with contextlib.suppress(Exception):
            candidate = os.getpgid(proc.pid)
            if candidate > 0 and candidate != os.getpgrp():
                process_group = candidate

    try:
        if process_group is not None:
            os.killpg(process_group, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return True
    except Exception as exc:
        _append_train_log(f"⚠ Could not request a graceful training stop: {exc}")

    try:
        proc.wait(timeout=max(0.1, float(graceful_timeout)))
        return True
    except subprocess.TimeoutExpired:
        _append_train_log("⚠ Training did not stop gracefully; forcing its process group to exit.")

    try:
        if process_group is not None:
            os.killpg(process_group, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        return True
    except Exception as exc:
        _append_train_log(f"⚠ Could not force the training process to stop: {exc}")

    try:
        proc.wait(timeout=max(0.1, float(kill_timeout)))
    except subprocess.TimeoutExpired:
        return False
    return proc.poll() is not None


def _start_training_thread(
    safe_word: str,
    language: str,
    auto_run: bool,
    training_lock,
    tts_mode: str = DEFAULT_SERVER_TTS_MODE,
    english_accent: str = DEFAULT_SERVER_ENGLISH_ACCENT,
) -> threading.Thread:
    global TRAINING_THREAD
    if TRAINING_SHUTDOWN_EVENT.is_set():
        raise RuntimeError("WakeWord Trainer is shutting down.")

    thread = threading.Thread(
        target=_run_training_background,
        args=(safe_word, language, auto_run, training_lock, tts_mode, english_accent),
        daemon=True,
        name="wake-word-training",
    )
    with TRAINING_RUNTIME_LOCK:
        if TRAINING_SHUTDOWN_EVENT.is_set():
            raise RuntimeError("WakeWord Trainer is shutting down.")
        if TRAINING_THREAD is not None and TRAINING_THREAD.is_alive():
            raise RuntimeError("Training is already running.")
        TRAINING_STOP_EVENT.clear()
        TRAINING_THREAD = thread
    try:
        thread.start()
    except Exception:
        with TRAINING_RUNTIME_LOCK:
            if TRAINING_THREAD is thread:
                TRAINING_THREAD = None
        raise
    return thread


def _stop_current_training(timeout: float = 20.0) -> bool:
    """Stop one training run without putting the server into shutdown mode."""

    TRAINING_STOP_EVENT.set()
    with TRAINING_RUNTIME_LOCK:
        proc = TRAINING_PROCESS
        thread = TRAINING_THREAD

    stopped = True
    if proc is not None:
        stopped = _terminate_training_process_tree(
            proc,
            graceful_timeout=min(12.0, max(1.0, float(timeout))),
        )
    if thread is not None and thread is not threading.current_thread() and thread.is_alive():
        thread.join(timeout=max(0.1, float(timeout)))
        stopped = stopped and not thread.is_alive()
    if stopped:
        TRAINING_STOP_EVENT.clear()
    return stopped


def _stop_training_runtime(timeout: float = 20.0) -> bool:
    TRAINING_SHUTDOWN_EVENT.set()
    return _stop_current_training(timeout=timeout)


def _start_auto_training() -> Dict[str, Any]:
    if TRAINING_SHUTDOWN_EVENT.is_set() or TRAINING_STOP_EVENT.is_set():
        return {"ok": False, "error": "WakeWord Trainer is shutting down."}
    with AUTO_TRAIN_LOCK:
        config = dict(AUTO_TRAIN_CONFIG)
    wake_phrase = str(config.get("wake_phrase") or "").strip()
    if not wake_phrase:
        return {"ok": False, "error": "Auto Training needs a wake phrase."}
    if not Path(TRAIN_SCRIPT).exists():
        return {"ok": False, "error": f"TRAIN_SCRIPT not found: {TRAIN_SCRIPT}"}

    safe_word = safe_name(wake_phrase)
    language = str(config.get("language") or DEFAULT_LANGUAGE)
    english_accent = normalize_english_accent(
        config.get("english_accent"), language
    )
    available_languages = _available_languages()
    tts_mode = _resolve_tts_mode_for_language(
        DEFAULT_SERVER_TTS_MODE, language, available_languages
    )
    with DATA_MANAGEMENT_LOCK:
        with STATE_LOCK:
            if STATE["training"]["running"]:
                return {"ok": False, "error": "Training already running."}
            try:
                training_lock = _try_acquire_training_run_lock()
            except Exception as exc:
                return {"ok": False, "error": f"Could not acquire the training lock: {exc}"}
            if training_lock is None:
                return {
                    "ok": False,
                    "error": "Training is already running in another trainer process.",
                    "code": "TRAINING_LOCKED",
                }
            STATE["raw_phrase"] = wake_phrase
            STATE["safe_word"] = safe_word
            STATE["language"] = language
            STATE["english_accent"] = english_accent
            STATE["tts_mode"] = tts_mode
            STATE["training"]["running"] = True
    try:
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["last_train_started_at"] = _iso_now()
            AUTO_TRAIN_RUNTIME["training_pending_consumed"] = int(
                AUTO_TRAIN_STATE.get("pending_negative_count") or 0
            )
            _save_auto_train_state_locked()
        _start_training_thread(
            safe_word,
            language,
            True,
            training_lock,
            tts_mode=tts_mode,
            english_accent=english_accent,
        )
    except Exception as exc:
        with STATE_LOCK:
            STATE["training"]["running"] = False
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_RUNTIME["training_pending_consumed"] = 0
        _release_training_run_lock(training_lock)
        return {"ok": False, "error": f"Could not start training: {exc}"}
    return {
        "ok": True,
        "started": True,
        "safe_word": safe_word,
        "language": language,
        "english_accent": english_accent,
        "tts_mode": tts_mode,
    }


def _maybe_run_scheduled_auto_training() -> None:
    with AUTO_TRAIN_LOCK:
        if not AUTO_TRAIN_CONFIG.get("enabled"):
            return
        schedule_hours = int(AUTO_TRAIN_CONFIG.get("schedule_hours") or 0)
        if schedule_hours <= 0:
            return
        next_run = _parse_iso_datetime(AUTO_TRAIN_STATE.get("next_run_at"))
        if next_run is None:
            _schedule_next_auto_run_locked()
            return
        now = _utc_now()
        if now < next_run:
            return
        pending = int(AUTO_TRAIN_STATE.get("pending_negative_count") or 0)
        minimum = int(AUTO_TRAIN_CONFIG.get("minimum_new_negatives") or 1)
        if pending < minimum:
            _schedule_next_auto_run_locked(from_time=now)
            return
    result = _start_auto_training()
    with AUTO_TRAIN_LOCK:
        if result.get("started"):
            _schedule_next_auto_run_locked()
        else:
            AUTO_TRAIN_STATE["next_run_at"] = (_utc_now() + timedelta(minutes=10)).isoformat()
            _save_auto_train_state_locked()


def _auto_train_worker_loop() -> None:
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_RUNTIME["scheduler_running"] = True
    _queue_pending_auto_reviews()
    try:
        while not AUTO_TRAIN_STOP_EVENT.is_set():
            try:
                file_name = AUTO_TRAIN_REVIEW_QUEUE.get_nowait()
            except queue.Empty:
                file_name = ""
            if file_name:
                try:
                    with DATA_MANAGEMENT_LOCK:
                        _auto_review_capture(file_name)
                finally:
                    with AUTO_TRAIN_LOCK:
                        AUTO_TRAIN_QUEUED_FILES.discard(file_name)
                    AUTO_TRAIN_REVIEW_QUEUE.task_done()
            _maybe_run_scheduled_auto_training()
            AUTO_TRAIN_WAKE_EVENT.wait(1.0)
            AUTO_TRAIN_WAKE_EVENT.clear()
    finally:
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_RUNTIME["scheduler_running"] = False


def _start_auto_train_worker() -> None:
    global AUTO_TRAIN_WORKER
    with AUTO_TRAIN_LOCK:
        if AUTO_TRAIN_WORKER is not None and AUTO_TRAIN_WORKER.is_alive():
            return
        AUTO_TRAIN_STOP_EVENT.clear()
        AUTO_TRAIN_WORKER = threading.Thread(
            target=_auto_train_worker_loop,
            name="auto-train-worker",
            daemon=True,
        )
        AUTO_TRAIN_WORKER.start()


def _stop_auto_train_worker(timeout: float = 5.0) -> bool:
    global AUTO_TRAIN_WORKER
    AUTO_TRAIN_STOP_EVENT.set()
    AUTO_TRAIN_WAKE_EVENT.set()
    worker = AUTO_TRAIN_WORKER
    if (
        worker is not None
        and worker is not threading.current_thread()
        and worker.is_alive()
    ):
        worker.join(timeout=max(0.1, float(timeout)))
    stopped = worker is None or not worker.is_alive()
    if stopped:
        with AUTO_TRAIN_LOCK:
            if AUTO_TRAIN_WORKER is worker:
                AUTO_TRAIN_WORKER = None
    return stopped


def _sync_personal_samples_state() -> List[str]:
    takes = _list_personal_samples()
    with STATE_LOCK:
        STATE["takes"] = takes
        STATE["takes_received"] = len(takes)
    return takes


def _registered_language_family(language: Dict[str, Any]) -> str:
    family = str(language.get("family") or "").strip().lower()
    if family:
        return family
    code = str(language.get("code") or "").strip()
    return code.split("_", 1)[0].lower() if code else ""


def _register_language(
    languages: Dict[str, Dict[str, Any]],
    *,
    family: str,
    name: str,
    region: str = "",
    count: int = 1,
    engine: str = "",
):
    if not family:
        return
    entry = languages.setdefault(
        family,
        {
            "code": family,
            "label": f"{name} ({family})",
            "name": name,
            "voice_count": 0,
            "regions": [],
            "engines": [],
        },
    )
    entry["voice_count"] += count
    if region and region not in entry["regions"]:
        entry["regions"].append(region)
    if engine and engine not in entry["engines"]:
        entry["engines"].append(engine)


def _fetch_omnivoice_catalog() -> Dict[str, Dict[str, Any]] | None:
    request = URLRequest(
        OMNIVOICE_LANGUAGES_URL,
        headers={"User-Agent": "microWakeWord-Trainer/modern-tts-apple-v1"},
    )
    with urlopen(request, timeout=15) as response:
        entries = parse_omnivoice_catalog(response.read().decode("utf-8"))
    return entries or None


def _read_cached_omnivoice_catalog_file() -> Dict[str, Dict[str, Any]] | None:
    try:
        data = json.loads(OMNIVOICE_CATALOG_CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_cached_omnivoice_catalog_file(data: Dict[str, Dict[str, Any]]) -> None:
    try:
        OMNIVOICE_CATALOG_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        OMNIVOICE_CATALOG_CACHE_FILE.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_omnivoice_catalog() -> Dict[str, Dict[str, Any]]:
    now = time.time()
    with OMNIVOICE_CATALOG_LOCK:
        cached = OMNIVOICE_CATALOG_CACHE.get("entries")
        fetched_at = float(OMNIVOICE_CATALOG_CACHE.get("fetched_at") or 0.0)
        if cached is not None and (now - fetched_at) < OMNIVOICE_CATALOG_CACHE_TTL_SECONDS:
            return cached

    disk_cached = _read_cached_omnivoice_catalog_file()
    try:
        fresh = _fetch_omnivoice_catalog()
    except Exception:
        fresh = None

    selected = fresh or disk_cached or {
        code: {"name": name, "iso_639_3": "", "duration_hours": 0.0}
        for code, name in COMMON_OMNIVOICE_LANGUAGES.items()
    }
    with OMNIVOICE_CATALOG_LOCK:
        OMNIVOICE_CATALOG_CACHE["entries"] = selected
        OMNIVOICE_CATALOG_CACHE["fetched_at"] = now
    if fresh:
        _write_cached_omnivoice_catalog_file(fresh)
    return selected


def _fetch_piper_catalog() -> Dict[str, Any] | None:
    req = URLRequest(
        PIPER_VOICES_INDEX_URL,
        headers={"User-Agent": "microWakeWord-Trainer/1.0"},
    )
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data if isinstance(data, dict) else None


def _read_cached_piper_catalog_file() -> Dict[str, Any] | None:
    try:
        if not PIPER_CATALOG_CACHE_FILE.exists():
            return None
        data = json.loads(PIPER_CATALOG_CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_cached_piper_catalog_file(data: Dict[str, Any]):
    try:
        PIPER_CATALOG_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PIPER_CATALOG_CACHE_FILE.write_text(
            json.dumps(data, ensure_ascii=True),
            encoding="utf-8",
        )
    except Exception:
        pass


def _load_piper_catalog() -> Dict[str, Any] | None:
    now = time.time()
    with PIPER_CATALOG_LOCK:
        cached = PIPER_CATALOG_CACHE.get("entries")
        fetched_at = float(PIPER_CATALOG_CACHE.get("fetched_at") or 0.0)
        if cached is not None and (now - fetched_at) < PIPER_CATALOG_CACHE_TTL_SECONDS:
            return cached

    disk_cached = _read_cached_piper_catalog_file()

    try:
        fresh = _fetch_piper_catalog()
    except Exception:
        fresh = None

    with PIPER_CATALOG_LOCK:
        if fresh is not None:
            PIPER_CATALOG_CACHE["entries"] = fresh
            PIPER_CATALOG_CACHE["fetched_at"] = now
            _write_cached_piper_catalog_file(fresh)
            return fresh
        if PIPER_CATALOG_CACHE.get("entries") is not None:
            return PIPER_CATALOG_CACHE.get("entries")
        if disk_cached is not None:
            PIPER_CATALOG_CACHE["entries"] = disk_cached
            PIPER_CATALOG_CACHE["fetched_at"] = now
            return disk_cached
        PIPER_CATALOG_CACHE["entries"] = {}
        PIPER_CATALOG_CACHE["fetched_at"] = now
        return PIPER_CATALOG_CACHE.get("entries")


def _available_languages() -> List[Dict[str, Any]]:
    languages: Dict[str, Dict[str, Any]] = {}
    omnivoice_catalog = _load_omnivoice_catalog()

    for code, metadata in omnivoice_catalog.items():
        if not isinstance(metadata, dict):
            continue
        _register_language(
            languages,
            family=code,
            name=str(metadata.get("name") or code.upper()),
            count=0,
            engine=ENGINE_OMNIVOICE,
        )

    for alias, catalog_code in OMNIVOICE_LANGUAGE_ALIASES.items():
        metadata = omnivoice_catalog.get(catalog_code) or {}
        _register_language(
            languages,
            family=alias,
            name=COMMON_OMNIVOICE_LANGUAGES.get(
                alias, str(metadata.get("name") or alias.upper())
            ),
            count=0,
            engine=ENGINE_OMNIVOICE,
        )

    for code, name in QWEN_LANGUAGES.items():
        _register_language(
            languages, family=code, name=name, count=0, engine=ENGINE_QWEN3
        )
    for code, name in MOSS_LANGUAGES.items():
        _register_language(
            languages, family=code, name=name, count=0, engine=ENGINE_MOSS
        )

    piper_english_model = PIPER_ROOT / "models" / "en_US-libritts_r-medium.pt"
    if piper_english_model.is_file():
        _register_language(
            languages,
            family="en",
            name="English",
            count=1,
            engine=ENGINE_PIPER,
        )

    if PIPER_VOICES_DIR.exists():
        for meta_path in sorted(PIPER_VOICES_DIR.glob("*.onnx.json")):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            language = data.get("language") or {}
            family = _registered_language_family(language)
            if not family:
                continue

            name = str(language.get("name_english") or language.get("name_native") or family.upper()).strip()
            region = str(language.get("country_english") or language.get("region") or "").strip()
            _register_language(
                languages,
                family=family,
                name=name,
                region=region,
                count=1,
                engine=ENGINE_PIPER,
            )

    catalog = _load_piper_catalog() or {}
    for entry in catalog.values():
        if not isinstance(entry, dict):
            continue
        language = entry.get("language") or {}
        family = _registered_language_family(language)
        if not family:
            continue
        name = str(language.get("name_english") or language.get("name_native") or family.upper()).strip()
        region = str(language.get("country_english") or language.get("region") or "").strip()
        _register_language(
            languages,
            family=family,
            name=name,
            region=region,
            count=0,
            engine=ENGINE_PIPER,
        )

    if "en" not in languages:
        _register_language(
            languages,
            family="en",
            name="English",
            count=0,
            engine=ENGINE_OMNIVOICE,
        )

    engine_order = (ENGINE_OMNIVOICE, ENGINE_QWEN3, ENGINE_MOSS, ENGINE_PIPER)
    display_names = {
        ENGINE_OMNIVOICE: "OmniVoice",
        ENGINE_QWEN3: "Qwen3",
        ENGINE_MOSS: "MOSS",
        ENGINE_PIPER: "Piper",
    }
    quality_labels = {
        "recommended": "Recommended",
        "supported": "Supported",
        "experimental": "Experimental",
        "legacy": "Legacy",
    }
    for entry in languages.values():
        entry["engines"] = [
            engine for engine in engine_order if engine in entry["engines"]
        ]
        entry["quality"] = quality_for_engines(entry["engines"])
        entry["engine_labels"] = [
            display_names[engine] for engine in entry["engines"]
        ]
        entry["label"] = (
            f"{entry['name']} ({entry['code']}) — "
            f"{quality_labels[entry['quality']]}"
        )

    ordered = [languages["en"]]
    ordered.extend(
        sorted(
            (entry for code, entry in languages.items() if code != "en"),
            key=lambda entry: (entry["name"].lower(), entry["code"]),
        )
    )
    return ordered


def _normalize_language(language: str | None) -> str:
    requested = (
        (language or DEFAULT_LANGUAGE).strip().lower().replace("-", "_")
        or DEFAULT_LANGUAGE
    )
    available_codes = {item["code"] for item in _available_languages()}
    if requested in available_codes:
        return requested
    family = requested.split("_", 1)[0]
    if family in available_codes:
        return family
    if DEFAULT_LANGUAGE in available_codes:
        return DEFAULT_LANGUAGE
    return "en"


def _resolve_tts_mode_for_language(
    mode: Any,
    language: str,
    available_languages: List[Dict[str, Any]],
) -> str:
    selected = normalize_tts_mode(mode)
    entry = next(
        (item for item in available_languages if item.get("code") == language),
        {},
    )
    engines = set(entry.get("engines") or [])
    has_modern = bool(
        engines.intersection({ENGINE_OMNIVOICE, ENGINE_QWEN3, ENGINE_MOSS})
    )
    has_piper = ENGINE_PIPER in engines
    if selected == "piper" and not has_piper:
        return "modern" if has_modern else selected
    if selected in {"modern", "hybrid"} and not has_modern and has_piper:
        return "piper"
    if selected == "hybrid" and not has_piper:
        return "modern"
    return selected


def _catalog_voice_files(language_family: str) -> List[tuple[str, str]]:
    if not language_family or language_family == "en":
        return []

    downloads: Dict[str, str] = {}
    catalog = _load_piper_catalog() or {}
    for entry in catalog.values():
        if not isinstance(entry, dict):
            continue
        language = entry.get("language") or {}
        family = _registered_language_family(language)
        if family != language_family:
            continue
        files = entry.get("files") or {}
        for rel_path in files.keys():
            if not isinstance(rel_path, str):
                continue
            if not (rel_path.endswith(".onnx") or rel_path.endswith(".onnx.json")):
                continue
            downloads[Path(rel_path).name] = f"{PIPER_VOICES_ROOT_URL}/{rel_path}?download=true"

    return sorted(downloads.items(), key=lambda item: item[0])


def _download_to_path(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req = URLRequest(url, headers={"User-Agent": "microWakeWord-Trainer/1.0"})
    with urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp_path.replace(dest_path)


def _ensure_non_english_language_voices(language_family: str, log) -> Dict[str, int]:
    downloads = _catalog_voice_files(language_family)
    local_voices = sorted(PIPER_VOICES_DIR.glob(f"{language_family}_*.onnx")) if PIPER_VOICES_DIR.exists() else []
    if not downloads:
        if local_voices:
            log(f"===== Piper Voices ({language_family}) =====")
            log(f"→ Using {len(local_voices)} installed voice(s) for language '{language_family}'")
            return {
                "downloaded_files": 0,
                "existing_files": len(local_voices),
                "voices": len(local_voices),
            }
        raise RuntimeError(
            f"No Piper ONNX voices found for language '{language_family}' in the upstream catalog."
        )

    PIPER_VOICES_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_files = 0
    existing_files = 0
    voice_names = sorted(name for name, _ in downloads if name.endswith(".onnx"))

    log(f"===== Piper Voices ({language_family}) =====")
    log(f"→ Ensuring {len(voice_names)} voice(s) for language '{language_family}'")

    for file_name, url in downloads:
        dest_path = PIPER_VOICES_DIR / file_name
        if dest_path.exists() and dest_path.stat().st_size > 0:
            existing_files += 1
            continue
        log(f"→ Downloading {file_name}")
        _download_to_path(url, dest_path)
        downloaded_files += 1

    log(
        f"✓ Piper voices ready for '{language_family}' "
        f"({downloaded_files} file(s) downloaded, {existing_files} already present)"
    )
    return {
        "downloaded_files": downloaded_files,
        "existing_files": existing_files,
        "voices": len(voice_names),
    }


def _find_ffmpeg() -> str | None:
    candidates = [
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/opt/ffmpeg@7/bin/ffmpeg",
        "/opt/homebrew/opt/ffmpeg/bin/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _inspect_wav_bytes(data: bytes) -> Dict[str, Any] | None:
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = (frames / rate) if rate else 0.0
            return {
                "container": "wav",
                "sample_rate": rate,
                "channels": wf.getnchannels(),
                "sample_width_bits": wf.getsampwidth() * 8,
                "compression": wf.getcomptype(),
                "frames": frames,
                "duration_s": round(duration, 3),
            }
    except Exception:
        return None


def _is_target_wav(info: Dict[str, Any] | None) -> bool:
    return bool(
        info
        and info.get("container") == "wav"
        and info.get("sample_rate") == TARGET_SAMPLE_RATE
        and info.get("channels") == TARGET_CHANNELS
        and info.get("sample_width_bits") == TARGET_SAMPLE_WIDTH_BYTES * 8
        and info.get("compression") == "NONE"
        and info.get("frames", 0) > 0
    )


def _next_personal_sample_name(original_name: str) -> str:
    return _next_directory_sample_name(PERSONAL_DIR, "sample", original_name)


def _next_negative_sample_name(original_name: str) -> str:
    return _next_directory_sample_name(NEGATIVE_DIR, "negative", original_name)


def _next_captured_sample_name(original_name: str) -> str:
    return _next_directory_sample_name(CAPTURED_DIR, "captured", original_name)


def _next_directory_sample_name(directory: Path, prefix: str, original_name: str) -> str:
    current = _list_audio_samples(directory)
    next_index = 1
    for name in current:
        match = re.match(rf"{re.escape(prefix)}_(\d{{4}})", name)
        if match:
            next_index = max(next_index, int(match.group(1)) + 1)

    stem = safe_name(Path(original_name or "sample").stem)
    suffix = f"_{stem[:32]}" if stem and stem != "wakeword" else ""
    return f"{prefix}_{next_index:04d}{suffix}.wav"


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _parse_probability_history(value: Any) -> List[int]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        raw_values = value
    else:
        raw_values = str(value).split(",")
    history: List[int] = []
    for raw_value in raw_values:
        parsed = _parse_int(raw_value)
        if parsed is not None:
            history.append(parsed)
    return history


def _audio_sidecar_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(".json")


def _load_sidecar_json(audio_path: Path) -> Dict[str, Any]:
    sidecar = _audio_sidecar_path(audio_path)
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_sidecar_json(audio_path: Path, payload: Dict[str, Any]):
    _audio_sidecar_path(audio_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _remove_audio_with_sidecar(audio_path: Path):
    if audio_path.exists():
        audio_path.unlink()
    sidecar = _audio_sidecar_path(audio_path)
    if sidecar.exists():
        sidecar.unlink()


def _resolve_audio_path(directory: Path, file_name: str) -> Path:
    candidate = Path(file_name or "").name
    if not candidate or candidate != (file_name or "") or not candidate.endswith(".wav"):
        raise FileNotFoundError("Invalid audio file name.")
    path = (directory / candidate).resolve()
    if path.parent != directory.resolve() or not path.exists():
        raise FileNotFoundError("Audio file not found.")
    return path


def _format_hint_from_filename(original_name: str) -> Dict[str, Any]:
    suffix = (Path(original_name or "").suffix or "").lower().lstrip(".")
    return {
        "container": suffix or "unknown",
        "sample_rate": None,
        "channels": None,
        "sample_width_bits": None,
        "compression": None,
        "frames": None,
        "duration_s": None,
    }


def _normalize_audio_to_target_wav(data: bytes, original_name: str) -> bytes:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required to convert uploads that are not already 16 kHz mono 16-bit PCM WAV."
        )

    suffix = (Path(original_name or "").suffix or ".audio")
    with tempfile.TemporaryDirectory(prefix="mww_upload_") as tmpdir:
        src_path = Path(tmpdir) / f"source{suffix}"
        dst_path = Path(tmpdir) / "normalized.wav"
        src_path.write_bytes(data)

        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(src_path),
            "-vn",
            "-ac",
            str(TARGET_CHANNELS),
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(dst_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not dst_path.exists():
            err = (proc.stderr or proc.stdout or "ffmpeg conversion failed").strip()
            raise RuntimeError(err.splitlines()[-1] if err else "ffmpeg conversion failed")

        return dst_path.read_bytes()


def _boost_target_wav_bytes(
    data: bytes,
    *,
    target_peak_ratio: float = 0.88,
    target_rms_ratio: float | None = None,
    max_gain_ratio: float = 10.0,
    min_gain_ratio: float = 1.25,
    profile: str | None = None,
) -> tuple[bytes, Dict[str, Any]]:
    info = _inspect_wav_bytes(data) or {}
    if not _is_target_wav(info):
        return data, {"applied": False, "reason": "not_target_wav"}

    with wave.open(io.BytesIO(data), "rb") as wf:
        raw_frames = wf.readframes(wf.getnframes())

    if not raw_frames:
        return data, {"applied": False, "reason": "empty"}

    samples = array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder != "little":
        samples.byteswap()

    peak = max(abs(sample) for sample in samples) if samples else 0
    if peak <= 0:
        return data, {"applied": False, "reason": "silent", "peak_ratio": 0.0}

    peak_ratio = peak / 32767.0
    rms_ratio = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5 / 32767.0
    desired_peak = max(0.05, min(target_peak_ratio, 0.98))
    peak_limited_gain = desired_peak / peak_ratio
    target_gain = peak_limited_gain
    if target_rms_ratio is not None and rms_ratio > 0:
        target_gain = min(target_rms_ratio / rms_ratio, peak_limited_gain)
    gain_ratio = min(max_gain_ratio, target_gain)

    if gain_ratio < min_gain_ratio:
        return data, {
            "applied": False,
            "reason": "already_loud_enough",
            "peak_ratio": round(peak_ratio, 4),
            "rms_ratio": round(rms_ratio, 4),
            "gain_ratio": round(gain_ratio, 3),
            "gain_db": round(20.0 * log10(max(gain_ratio, 1e-9)), 2),
            "profile": profile or "",
        }

    boosted = array("h", (max(-32768, min(32767, int(round(sample * gain_ratio)))) for sample in samples))
    if sys.byteorder != "little":
        boosted.byteswap()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(TARGET_CHANNELS)
        wav.setsampwidth(TARGET_SAMPLE_WIDTH_BYTES)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes(boosted.tobytes())

    return buf.getvalue(), {
        "applied": True,
        "peak_ratio": round(peak_ratio, 4),
        "rms_ratio": round(rms_ratio, 4),
        "gain_ratio": round(gain_ratio, 3),
        "gain_db": round(20.0 * log10(max(gain_ratio, 1e-9)), 2),
        "profile": profile or "",
    }


def _build_audio_result_message(*, converted: bool, postprocess_info: Dict[str, Any] | None = None) -> str:
    message = (
        "Converted to 16 kHz mono 16-bit PCM WAV"
        if converted
        else "Already in the correct 16 kHz mono 16-bit PCM WAV format"
    )
    if postprocess_info and postprocess_info.get("applied"):
        message += f"; boosted {postprocess_info['gain_db']} dB for clearer captured playback"
    return message


def _ensure_captured_playback_ready(audio_path: Path, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    existing_postprocess = metadata.get("postprocess")
    if isinstance(existing_postprocess, dict) and existing_postprocess.get("profile") == CAPTURE_GAIN_PROFILE:
        return metadata

    with SAMPLES_LOCK:
        data = audio_path.read_bytes()
        final_bytes, postprocess_info = _boost_target_wav_bytes(
            data,
            target_peak_ratio=0.88,
            target_rms_ratio=0.06,
            max_gain_ratio=220.0,
            profile=CAPTURE_GAIN_PROFILE,
        )
        if postprocess_info.get("applied"):
            audio_path.write_bytes(final_bytes)
        if isinstance(existing_postprocess, dict):
            try:
                previous_gain = float(existing_postprocess.get("gain_ratio") or 1.0)
            except Exception:
                previous_gain = 1.0
            current_gain = float(postprocess_info.get("gain_ratio") or 1.0)
            total_gain = previous_gain * current_gain
            if previous_gain != 1.0:
                postprocess_info["gain_ratio"] = round(total_gain, 3)
                postprocess_info["gain_db"] = round(20.0 * log10(max(total_gain, 1e-9)), 2)
        metadata["postprocess"] = postprocess_info
        metadata["final_format"] = _inspect_wav_bytes(final_bytes) or metadata.get("final_format") or {}
        metadata["message"] = _build_audio_result_message(
            converted=bool(metadata.get("converted")),
            postprocess_info=postprocess_info,
        )
        _write_sidecar_json(audio_path, metadata)

    return metadata


def _save_audio_sample(
    data: bytes,
    original_name: str,
    *,
    target_dir: Path,
    out_name: str,
    postprocess_target_wav: Callable[[bytes], tuple[bytes, Dict[str, Any]]] | None = None,
) -> Dict[str, Any]:
    if not data:
        raise ValueError("Empty or invalid audio file.")

    original_info = _inspect_wav_bytes(data) or _format_hint_from_filename(original_name)
    normalized = _is_target_wav(original_info)
    final_bytes = data if normalized else _normalize_audio_to_target_wav(data, original_name)
    postprocess_info: Dict[str, Any] = {"applied": False}
    if postprocess_target_wav is not None:
        final_bytes, postprocess_info = postprocess_target_wav(final_bytes)
    final_info = _inspect_wav_bytes(final_bytes)

    if not _is_target_wav(final_info):
        raise ValueError("Uploaded audio could not be normalized to 16 kHz mono 16-bit PCM WAV.")

    with SAMPLES_LOCK:
        target_dir.mkdir(parents=True, exist_ok=True)
        final_name = out_name
        out_path = target_dir / final_name
        out_path.write_bytes(final_bytes)

    return {
        "saved_as": final_name,
        "converted": not normalized,
        "postprocess": postprocess_info,
        "original_name": original_name or final_name,
        "detected_format": original_info,
        "final_format": final_info,
        "message": _build_audio_result_message(
            converted=not normalized,
            postprocess_info=postprocess_info,
        ),
    }


def _save_personal_sample(data: bytes, original_name: str, out_name: str | None = None) -> Dict[str, Any]:
    return _save_audio_sample(
        data,
        original_name,
        target_dir=PERSONAL_DIR,
        out_name=out_name or _next_personal_sample_name(original_name),
    )


def _save_captured_sample(data: bytes, original_name: str, out_name: str | None = None) -> Dict[str, Any]:
    return _save_audio_sample(
        data,
        original_name,
        target_dir=CAPTURED_DIR,
        out_name=out_name or _next_captured_sample_name(original_name),
        postprocess_target_wav=lambda wav_data: _boost_target_wav_bytes(
            wav_data,
            target_peak_ratio=0.88,
            target_rms_ratio=0.06,
            max_gain_ratio=220.0,
            profile=CAPTURE_GAIN_PROFILE,
        ),
    )


def _pcm_s16le_to_wav_bytes(
    pcm_data: bytes,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    channels: int = TARGET_CHANNELS,
    sample_width_bytes: int = TARGET_SAMPLE_WIDTH_BYTES,
) -> bytes:
    if not pcm_data:
        raise ValueError("Captured audio payload was empty.")

    if sample_width_bytes <= 0:
        raise ValueError("Invalid sample width for PCM conversion.")

    frame_width = channels * sample_width_bytes
    if frame_width <= 0 or (len(pcm_data) % frame_width) != 0:
        raise ValueError("Captured PCM payload does not align to whole audio frames.")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width_bytes)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)
    return buf.getvalue()


def _captured_item_from_path(audio_path: Path) -> Dict[str, Any]:
    meta = _ensure_captured_playback_ready(audio_path, _load_sidecar_json(audio_path))
    stat = audio_path.stat()
    event_type = str(meta.get("event_type") or "captured").strip() or "captured"
    final_format = meta.get("final_format") or _inspect_wav_bytes(audio_path.read_bytes()) or {}
    return {
        "saved_as": audio_path.name,
        "original_name": meta.get("original_name") or audio_path.name,
        "source_device": meta.get("source_device") or "",
        "wake_word": meta.get("wake_word") or "",
        "event_type": event_type,
        "capture_label": str(meta.get("capture_label") or event_type.replace("_", " ").title()),
        "received_at": meta.get("received_at") or datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "captured_at": meta.get("captured_at") or "",
        "converted": bool(meta.get("converted")),
        "blocked_by_vad": bool(meta.get("blocked_by_vad")),
        "max_probability": meta.get("max_probability"),
        "average_probability": meta.get("average_probability"),
        "probability_cutoff": meta.get("probability_cutoff"),
        "peak_probability_cutoff": meta.get("peak_probability_cutoff"),
        "active_window_count": meta.get("active_window_count"),
        "min_active_windows": meta.get("min_active_windows"),
        "rise_score": meta.get("rise_score"),
        "vad_max_probability": meta.get("vad_max_probability"),
        "vad_average_probability": meta.get("vad_average_probability"),
        "detection_profile": meta.get("detection_profile") or "",
        "probability_history": meta.get("probability_history") or [],
        "detected_format": meta.get("detected_format") or {},
        "final_format": final_format,
        "postprocess": meta.get("postprocess") or {},
        "message": meta.get("message") or "",
        "notes": meta.get("notes") or "",
        "review_status": meta.get("review_status") or "pending",
        "transcript": meta.get("transcript") or "",
        "transcribed_at": meta.get("transcribed_at") or "",
        "auto_review_status": meta.get("auto_review_status") or "",
        "auto_review_reason": meta.get("auto_review_reason") or "",
        "auto_review_error": meta.get("auto_review_error") or "",
        "auto_review_guided_transcript": meta.get("auto_review_guided_transcript") or "",
        "auto_review_phrase_similarity": meta.get("auto_review_phrase_similarity"),
        "auto_review_match_method": meta.get("auto_review_match_method") or "",
        "size_bytes": stat.st_size,
        "audio_url": f"/api/audio/captured/{audio_path.name}",
    }


def _list_captured_items() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    CAPTURED_DIR.mkdir(parents=True, exist_ok=True)
    for audio_path in sorted(CAPTURED_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items.append(_captured_item_from_path(audio_path))
        except Exception:
            continue
    return items


def _sample_item_from_path(audio_path: Path, bucket: str) -> Dict[str, Any]:
    meta = _load_sidecar_json(audio_path)
    stat = audio_path.stat()
    final_format = meta.get("final_format") or meta.get("detected_format") or _inspect_wav_bytes(audio_path.read_bytes()) or {}
    return {
        "bucket": bucket,
        "saved_as": audio_path.name,
        "original_name": meta.get("original_name") or audio_path.name,
        "wake_word": meta.get("wake_word") or "",
        "event_type": meta.get("event_type") or "",
        "review_status": meta.get("review_status") or "",
        "received_at": meta.get("received_at") or "",
        "reviewed_at": meta.get("reviewed_at") or "",
        "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "converted": bool(meta.get("converted")),
        "trimmed": bool(meta.get("trimmed")),
        "source_file": meta.get("source_file") or "",
        "final_format": final_format,
        "message": meta.get("message") or "",
        "transcript": meta.get("transcript") or "",
        "transcribed_at": meta.get("transcribed_at") or "",
        "auto_review_guided_transcript": meta.get("auto_review_guided_transcript") or "",
        "auto_review_stt_engine": meta.get("auto_review_stt_engine") or "",
        "auto_review_stt_model": meta.get("auto_review_stt_model") or "",
        "auto_negative": bool(meta.get("auto_negative")),
        "auto_positive": bool(meta.get("auto_positive")),
        "auto_review_reason": meta.get("auto_review_reason") or "",
        "size_bytes": stat.st_size,
        "audio_url": f"/api/audio/{bucket}/{audio_path.name}",
    }


def _list_sample_items(directory: Path, bucket: str) -> List[Dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    items: List[Dict[str, Any]] = []
    for audio_path in sorted(directory.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items.append(_sample_item_from_path(audio_path, bucket))
        except Exception:
            continue
    # Untrimmed first (stable sort preserves mtime order within each group)
    items.sort(key=lambda x: x.get("trimmed", False))
    return items


def _samples_payload() -> Dict[str, Any]:
    takes = _sync_personal_samples_state()
    personal_items = _list_sample_items(PERSONAL_DIR, "personal")
    negative_items = _list_sample_items(NEGATIVE_DIR, "negative")
    return {
        "ok": True,
        "personal": personal_items,
        "negative": negative_items,
        "personal_count": len(personal_items),
        "negative_count": len(negative_items),
        "takes_received": len(takes),
    }


def _move_captured_audio(file_name: str, target_dir: Path, *, target_prefix: str, review_status: str) -> Dict[str, Any]:
    with SAMPLES_LOCK:
        src_path = _resolve_audio_path(CAPTURED_DIR, file_name)
        metadata = _load_sidecar_json(src_path)
        original_name = str(metadata.get("original_name") or src_path.name)
        if target_prefix == "sample":
            target_name = _next_personal_sample_name(original_name)
        else:
            target_name = _next_negative_sample_name(original_name)

        target_dir.mkdir(parents=True, exist_ok=True)
        dst_path = target_dir / target_name
        src_path.replace(dst_path)

        metadata["review_status"] = review_status
        metadata["reviewed_at"] = datetime.now(timezone.utc).isoformat()
        metadata["saved_as"] = target_name
        _write_sidecar_json(dst_path, metadata)

        stale_sidecar = _audio_sidecar_path(src_path)
        if stale_sidecar.exists():
            stale_sidecar.unlink()

    takes = _sync_personal_samples_state()
    return {
        "saved_as": target_name,
        "captured_remaining": len(_list_captured_sample_names()),
        "negative_count": len(_list_negative_samples()),
        "takes_received": len(takes),
    }


def _append_train_log(line: str):
    line = (line or "").rstrip("\n")
    with STATE_LOCK:
        buf: List[str] = STATE["training"]["log_lines"]
        buf.append(line)
        if len(buf) > 250:
            del buf[: len(buf) - 250]


def _run_training_background(
    safe_word: str,
    language: str,
    auto_run: bool = False,
    training_lock=None,
    tts_mode: str = DEFAULT_SERVER_TTS_MODE,
    english_accent: str = DEFAULT_SERVER_ENGLISH_ACCENT,
):
    global TRAINING_PROCESS, TRAINING_THREAD
    language = (language or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE
    tts_mode = normalize_tts_mode(tts_mode)
    english_accent = normalize_english_accent(english_accent, language)
    rc = 999
    proc: subprocess.Popen | None = None

    with STATE_LOCK:
        raw_phrase = str(STATE.get("raw_phrase") or "").strip()
        STATE["training"]["running"] = True
        STATE["training"]["exit_code"] = None
        STATE["training"]["log_lines"] = []
        STATE["training"]["safe_word"] = safe_word
        log_path = str(DATA_DIR / "recorder_training.log")
        STATE["training"]["log_path"] = log_path

    training_phrase = raw_phrase or safe_word
    cmd = ["bash", TRAIN_SCRIPT, training_phrase]

    _append_train_log(f"→ Running: {' '.join(cmd)}")
    _append_train_log(f"→ Language: {language}")
    _append_train_log(f"→ TTS mode: {tts_mode}")
    if language == "en":
        _append_train_log(f"→ English accent emphasis: {english_accent}")

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if language != "en" and tts_mode == "piper":
            _ensure_non_english_language_voices(language, _append_train_log)
        elif language != "en" and tts_mode == "hybrid":
            try:
                _ensure_non_english_language_voices(language, _append_train_log)
            except Exception as error:
                _append_train_log(
                    f"⚠️ Piper is unavailable for hybrid mode; using modern TTS only: {error}"
                )

        with open(log_path, "w", encoding="utf-8") as lf:
            env = os.environ.copy()
            env["MWW_LANGUAGE"] = language
            env["MWW_TTS_MODE"] = tts_mode
            env["MWW_ENGLISH_ACCENT"] = english_accent
            env["MWW_TTS_VOICE_COUNT"] = str(DEFAULT_TTS_VOICE_COUNT)
            env["MWW_ARTIFACT_SLUG"] = safe_word
            env["WAKEWORD_TRAINER_SUPPORT_DIR"] = str(SUPPORT_DIR)
            env["WAKEWORD_TRAINER_DATA_DIR"] = str(DATA_DIR)
            env["TRAINED_WAKE_WORDS_DIR"] = str(TRAINED_WAKE_WORDS_DIR)
            proc = subprocess.Popen(
                cmd,
                cwd=str(DATA_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                pass_fds=(training_lock.fileno(),) if training_lock is not None else (),
                start_new_session=(os.name == "posix"),
            )
            with TRAINING_RUNTIME_LOCK:
                TRAINING_PROCESS = proc
            if TRAINING_SHUTDOWN_EVENT.is_set() or TRAINING_STOP_EVENT.is_set():
                reason = "Trainer shutdown" if TRAINING_SHUTDOWN_EVENT.is_set() else "Session stop"
                _append_train_log(f"→ {reason} requested; stopping the active training run.")
                _terminate_training_process_tree(proc)
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    lf.write(line)
                    lf.flush()
                    _append_train_log(line)
            finally:
                with contextlib.suppress(Exception):
                    proc.stdout.close()

            rc = proc.wait()

        if (TRAINING_SHUTDOWN_EVENT.is_set() or TRAINING_STOP_EVENT.is_set()) and rc != 0:
            reason = "Trainer shutdown" if TRAINING_SHUTDOWN_EVENT.is_set() else "session stop"
            _append_train_log(f"→ Training stopped for {reason} (exit_code={rc})")
        else:
            _append_train_log(f"✓ Training finished (exit_code={rc})")
        with STATE_LOCK:
            STATE["training"]["exit_code"] = rc

    except Exception as e:
        _append_train_log(f"✗ Training crashed: {e!r}")
        with STATE_LOCK:
            STATE["training"]["exit_code"] = 999

    try:
        if auto_run:
            with AUTO_TRAIN_LOCK:
                AUTO_TRAIN_STATE["last_train_finished_at"] = _iso_now()
                AUTO_TRAIN_STATE["last_train_exit_code"] = rc
                if rc == 0 and not TRAINING_SHUTDOWN_EVENT.is_set() and not TRAINING_STOP_EVENT.is_set():
                    consumed = int(AUTO_TRAIN_RUNTIME.get("training_pending_consumed") or 0)
                    AUTO_TRAIN_STATE["pending_negative_count"] = max(
                        0,
                        int(AUTO_TRAIN_STATE.get("pending_negative_count") or 0) - consumed,
                    )
                AUTO_TRAIN_RUNTIME["training_pending_consumed"] = 0
                _save_auto_train_state_locked()
            if rc == 0 and not TRAINING_SHUTDOWN_EVENT.is_set() and not TRAINING_STOP_EVENT.is_set():
                _append_train_log("→ Publishing the newly trained wake word to Tater and all satellites")
                notify_result = _notify_tater_satellites(safe_word)
                if notify_result.get("ok"):
                    if notify_result.get("skipped"):
                        _append_train_log("→ Wake-word publish skipped (disabled in Auto Training)")
                    else:
                        count = notify_result.get("count")
                        suffix = f" ({count} connected)" if count is not None else ""
                        _append_train_log(f"✓ New wake word activated through Tater{suffix}")
                else:
                    _append_train_log(f"✗ Tater wake-word activation failed: {notify_result.get('error')}")
    finally:
        with TRAINING_RUNTIME_LOCK:
            if TRAINING_PROCESS is proc:
                TRAINING_PROCESS = None
            if TRAINING_THREAD is threading.current_thread():
                TRAINING_THREAD = None
        with STATE_LOCK:
            STATE["training"]["running"] = False
        _release_training_run_lock(training_lock)
        if not TRAINING_SHUTDOWN_EVENT.is_set():
            TRAINING_STOP_EVENT.clear()


# -------------------- Routes --------------------
@app.on_event("startup")
def start_auto_train_worker_event():
    TRAINING_SHUTDOWN_EVENT.clear()
    _start_auto_train_worker()


@app.on_event("shutdown")
def stop_auto_train_worker_event():
    worker_stopped = _stop_auto_train_worker(timeout=5.0)
    training_stopped = _stop_training_runtime(timeout=20.0)
    if not worker_stopped:
        _append_train_log("⚠ Auto-training worker did not stop before shutdown.")
    if not training_stopped:
        _append_train_log("⚠ Training worker did not stop before shutdown.")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h3>Missing UI</h3><p>Create <code>static/index.html</code>.</p>",
            status_code=500,
        )
    return HTMLResponse(
        html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/auto_train")
def auto_train_status(request: Request):
    payload = _auto_train_status_payload()
    payload["ok"] = True
    payload["advertised_base_url"] = _advertised_base_url(request)
    payload["stt_backend"] = payload["config"].get("stt_engine")
    return payload


@app.put("/api/auto_train")
def update_auto_train(payload: Dict[str, Any] = None):
    incoming = dict(payload or {})
    for protected_key in (
        "tater_link_token",
        "tater_link_id",
        "tater_linked_at",
        "tater_link_tater_name",
    ):
        incoming.pop(protected_key, None)
    with AUTO_TRAIN_LOCK:
        previous = dict(AUTO_TRAIN_CONFIG)
        try:
            normalized = _normalize_auto_train_config(incoming, base=previous)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        if normalized["enabled"] and not normalized["wake_phrase"]:
            return JSONResponse(
                {"ok": False, "error": "Enter the wake phrase before enabling Auto Training."},
                status_code=400,
            )
        AUTO_TRAIN_CONFIG.clear()
        AUTO_TRAIN_CONFIG.update(normalized)
        _save_auto_train_config_locked()
        schedule_changed = (
            previous.get("enabled") != normalized.get("enabled")
            or previous.get("schedule_hours") != normalized.get("schedule_hours")
        )
        if schedule_changed or not AUTO_TRAIN_STATE.get("next_run_at"):
            _schedule_next_auto_run_locked()
    if previous.get("stt_engine") != normalized.get("stt_engine"):
        _clear_stt_model_caches(keep_engine=normalized["stt_engine"])
    if normalized["enabled"]:
        queued = _queue_pending_auto_reviews()
        AUTO_TRAIN_WAKE_EVENT.set()
    else:
        queued = 0
    return {"ok": True, "queued": queued, **_auto_train_status_payload()}


@app.post("/api/tater_link/claim")
def tater_link_claim(payload: Dict[str, Any] = None):
    body = payload if isinstance(payload, dict) else {}
    try:
        return _claim_tater_link(
            body.get("tater_url"),
            body.get("pairing_code"),
        )
    except ValueError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)


@app.post("/api/tater_link/unlink")
def tater_link_unlink():
    return _unlink_tater()


@app.post("/api/auto_train/action")
def auto_train_action(payload: Dict[str, Any] = None):
    action = str((payload or {}).get("action") or "").strip().lower()
    if action == "review_now":
        with AUTO_TRAIN_LOCK:
            if not AUTO_TRAIN_CONFIG.get("enabled"):
                return JSONResponse({"ok": False, "error": "Enable Auto Training first."}, status_code=400)
        queued = _queue_pending_auto_reviews(force=True)
        AUTO_TRAIN_WAKE_EVENT.set()
        return {"ok": True, "queued": queued, **_auto_train_status_payload()}
    if action == "train_now":
        result = _start_auto_training()
        if not result.get("ok"):
            return JSONResponse(result, status_code=400)
        return {**result, **_auto_train_status_payload()}
    if action == "notify_now":
        result = _notify_tater_satellites()
        if not result.get("ok"):
            return JSONResponse(result, status_code=502)
        return {**result, **_auto_train_status_payload()}
    return JSONResponse({"ok": False, "error": "Unknown Auto Training action."}, status_code=400)


@app.post("/api/start_session")
def start_session(payload: Dict[str, Any]):
    with STATE_LOCK:
        active_safe_word = STATE.get("safe_word")
    if active_safe_word:
        return JSONResponse(
            {
                "ok": False,
                "error": f"Session '{active_safe_word}' is already active. Stop it before changing the wake phrase.",
                "code": "SESSION_ACTIVE",
            },
            status_code=409,
        )

    raw = (payload.get("phrase") or "").strip()
    if not raw:
        return JSONResponse({"ok": False, "error": "phrase is required"}, status_code=400)

    safe = safe_name(raw)

    speakers_total = int(payload.get("speakers_total") or SPEAKERS_TOTAL_DEFAULT)
    takes_per_speaker = int(payload.get("takes_per_speaker") or TAKES_PER_SPEAKER_DEFAULT)
    language = _normalize_language(payload.get("language"))
    available_languages = _available_languages()
    tts_mode = _resolve_tts_mode_for_language(
        payload.get("tts_mode", DEFAULT_SERVER_TTS_MODE),
        language,
        available_languages,
    )
    english_accent = normalize_english_accent(
        payload.get("english_accent", DEFAULT_SERVER_ENGLISH_ACCENT), language
    )

    speakers_total = max(1, min(10, speakers_total))
    takes_per_speaker = max(1, min(50, takes_per_speaker))

    with STATE_LOCK:
        STATE["raw_phrase"] = raw
        STATE["safe_word"] = safe
        STATE["language"] = language
        STATE["english_accent"] = english_accent
        STATE["tts_mode"] = tts_mode
        STATE["speakers_total"] = speakers_total
        STATE["takes_per_speaker"] = takes_per_speaker
        # do not interrupt training if running
    takes = _sync_personal_samples_state()

    return {
        "ok": True,
        "raw_phrase": raw,
        "safe_word": safe,
        "language": language,
        "english_accent": english_accent,
        "tts_mode": tts_mode,
        "speakers_total": speakers_total,
        "takes_per_speaker": takes_per_speaker,
        "takes_total": speakers_total * takes_per_speaker,
        "takes_received": len(takes),
        "takes": takes,
        "available_languages": available_languages,
        "available_english_accents": english_accent_options(),
    }


@app.post("/api/stop_session")
def stop_session():
    with STATE_LOCK:
        had_session = bool(STATE.get("safe_word"))
        training_running = bool(STATE["training"].get("running"))

    if training_running and not _stop_current_training(timeout=20.0):
        return JSONResponse(
            {
                "ok": False,
                "error": "Training did not stop cleanly; the session remains active.",
                "code": "TRAINING_STOP_TIMEOUT",
            },
            status_code=500,
        )

    takes = _sync_personal_samples_state()
    available_languages = _available_languages()
    with STATE_LOCK:
        STATE["raw_phrase"] = None
        STATE["safe_word"] = None
        STATE["training"]["safe_word"] = None
        training = dict(STATE["training"])
        language = _normalize_language(STATE["language"])
        english_accent = normalize_english_accent(
            STATE.get("english_accent"), language
        )
        tts_mode = normalize_tts_mode(STATE.get("tts_mode"))
    return {
        "ok": True,
        "session_stopped": had_session,
        "training_stopped": training_running,
        "raw_phrase": None,
        "safe_word": None,
        "language": language,
        "english_accent": english_accent,
        "tts_mode": tts_mode,
        "takes_received": len(takes),
        "takes": list(takes),
        "training": training,
        "available_languages": available_languages,
        "available_english_accents": english_accent_options(),
    }


@app.get("/api/session")
def get_session():
    takes = _sync_personal_samples_state()
    available_languages = _available_languages()
    with STATE_LOCK:
        current_language = _normalize_language(STATE["language"])
        current_tts_mode = normalize_tts_mode(STATE.get("tts_mode"))
        current_english_accent = normalize_english_accent(
            STATE.get("english_accent"), current_language
        )
        STATE["language"] = current_language
        STATE["english_accent"] = current_english_accent
        STATE["tts_mode"] = current_tts_mode
        return {
            "ok": True,
            "raw_phrase": STATE["raw_phrase"],
            "safe_word": STATE["safe_word"],
            "language": current_language,
            "english_accent": current_english_accent,
            "tts_mode": current_tts_mode,
            "speakers_total": STATE["speakers_total"],
            "takes_per_speaker": STATE["takes_per_speaker"],
            "takes_received": len(takes),
            "takes": list(takes),
            "training": dict(STATE["training"]),
            "available_languages": available_languages,
            "available_english_accents": english_accent_options(),
        }


@app.post("/api/upload_take")
async def upload_take(
    speaker_index: int = Form(...),
    take_index: int = Form(...),
    file: UploadFile = File(...),
):
    with STATE_LOCK:
        safe_word = STATE["safe_word"]
        speakers_total = int(STATE["speakers_total"])
        takes_per_speaker = int(STATE["takes_per_speaker"])

    if not safe_word:
        return JSONResponse({"ok": False, "error": "No active session. Call /api/start_session first."}, status_code=400)

    if speaker_index < 1 or speaker_index > speakers_total:
        return JSONResponse({"ok": False, "error": f"speaker_index must be 1..{speakers_total}"}, status_code=400)

    if take_index < 1 or take_index > takes_per_speaker:
        return JSONResponse({"ok": False, "error": f"take_index must be 1..{takes_per_speaker}"}, status_code=400)

    out_name = f"speaker{speaker_index:02d}_take{take_index:02d}.wav"

    data = await file.read()
    try:
        result = _save_personal_sample(data, file.filename or out_name, out_name=out_name)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    takes = _sync_personal_samples_state()
    return {"ok": True, **result, "takes_received": len(takes)}


@app.post("/api/upload_personal_sample")
async def upload_personal_sample(file: UploadFile = File(...)):
    with STATE_LOCK:
        safe_word = STATE["safe_word"]

    if not safe_word:
        return JSONResponse({"ok": False, "error": "No active session. Call /api/start_session first."}, status_code=400)

    data = await file.read()
    try:
        result = _save_personal_sample(data, file.filename or "sample")
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    takes = _sync_personal_samples_state()
    return {"ok": True, **result, "takes_received": len(takes)}


@app.post("/api/upload_captured_audio")
async def upload_captured_audio(
    file: UploadFile = File(...),
    source_device: str | None = Form(None),
    wake_word: str | None = Form(None),
    event_type: str | None = Form(None),
    captured_at: str | None = Form(None),
    blocked_by_vad: str | None = Form(None),
    max_probability: str | None = Form(None),
    average_probability: str | None = Form(None),
    notes: str | None = Form(None),
    metadata_json: str | None = Form(None),
):
    data = await file.read()
    try:
        result = _save_captured_sample(data, file.filename or "captured")
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    extra_meta: Dict[str, Any] = {}
    if metadata_json:
        try:
            parsed = json.loads(metadata_json)
            if isinstance(parsed, dict):
                extra_meta = parsed
        except Exception:
            return JSONResponse({"ok": False, "error": "metadata_json must be a JSON object"}, status_code=400)

    with STATE_LOCK:
        current_safe_word = STATE.get("safe_word")

    audio_path = CAPTURED_DIR / result["saved_as"]
    sidecar = {
        **extra_meta,
        "saved_as": result["saved_as"],
        "original_name": result["original_name"],
        "source_device": source_device or extra_meta.get("source_device") or "",
        "wake_word": wake_word or extra_meta.get("wake_word") or current_safe_word or "",
        "event_type": (event_type or extra_meta.get("event_type") or "captured").strip() or "captured",
        "capture_label": extra_meta.get("capture_label") or "",
        "captured_at": captured_at or extra_meta.get("captured_at") or "",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "blocked_by_vad": _parse_bool(extra_meta.get("blocked_by_vad") if blocked_by_vad is None else blocked_by_vad),
        "max_probability": _parse_float(extra_meta.get("max_probability") if max_probability is None else max_probability),
        "average_probability": _parse_float(
            extra_meta.get("average_probability") if average_probability is None else average_probability
        ),
        "probability_cutoff": _parse_int(extra_meta.get("probability_cutoff")),
        "peak_probability_cutoff": _parse_int(extra_meta.get("peak_probability_cutoff")),
        "active_window_count": _parse_int(extra_meta.get("active_window_count")),
        "min_active_windows": _parse_int(extra_meta.get("min_active_windows")),
        "rise_score": _parse_int(extra_meta.get("rise_score")),
        "vad_max_probability": _parse_int(extra_meta.get("vad_max_probability")),
        "vad_average_probability": _parse_int(extra_meta.get("vad_average_probability")),
        "detection_profile": str(extra_meta.get("detection_profile") or "").strip(),
        "probability_history": _parse_probability_history(extra_meta.get("probability_history")),
        "notes": notes or extra_meta.get("notes") or "",
        "converted": result["converted"],
        "detected_format": result["detected_format"],
        "final_format": result["final_format"],
        "postprocess": result["postprocess"],
        "message": result["message"],
        "review_status": "pending",
    }
    _write_sidecar_json(audio_path, sidecar)
    with AUTO_TRAIN_LOCK:
        auto_review_config = dict(AUTO_TRAIN_CONFIG)
    if auto_review_config.get("enabled") and _captured_event_is_auto_reviewable(sidecar, auto_review_config):
        _queue_auto_review(audio_path.name)

    return {
        "ok": True,
        "item": _captured_item_from_path(audio_path),
        "captured_count": len(_list_captured_sample_names()),
    }


@app.post("/api/upload_captured_audio_raw")
async def upload_captured_audio_raw(
    request: Request,
    x_audio_format: str | None = Header(default=None),
    x_original_name: str | None = Header(default=None),
    x_source_device: str | None = Header(default=None),
    x_wake_word: str | None = Header(default=None),
    x_event_type: str | None = Header(default=None),
    x_captured_at: str | None = Header(default=None),
    x_blocked_by_vad: str | None = Header(default=None),
    x_max_probability: str | None = Header(default=None),
    x_average_probability: str | None = Header(default=None),
    x_probability_cutoff: str | None = Header(default=None),
    x_peak_probability_cutoff: str | None = Header(default=None),
    x_active_windows: str | None = Header(default=None),
    x_min_active_windows: str | None = Header(default=None),
    x_rise_score: str | None = Header(default=None),
    x_vad_max_probability: str | None = Header(default=None),
    x_vad_average_probability: str | None = Header(default=None),
    x_detection_profile: str | None = Header(default=None),
    x_probability_history: str | None = Header(default=None),
    x_notes: str | None = Header(default=None),
):
    raw_data = await request.body()
    audio_format = (x_audio_format or "wav").strip().lower()

    try:
        if audio_format == "pcm_s16le":
            data = _pcm_s16le_to_wav_bytes(raw_data)
            original_name = x_original_name or "captured.raw.wav"
        elif audio_format in {"wav", "audio/wav", "audio/x-wav"}:
            data = raw_data
            original_name = x_original_name or "captured.wav"
        else:
            raise ValueError(f"Unsupported x-audio-format '{audio_format}'.")

        result = _save_captured_sample(data, original_name)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    with STATE_LOCK:
        current_safe_word = STATE.get("safe_word")

    audio_path = CAPTURED_DIR / result["saved_as"]
    sidecar = {
        "saved_as": result["saved_as"],
        "original_name": result["original_name"],
        "source_device": x_source_device or "",
        "wake_word": x_wake_word or current_safe_word or "",
        "event_type": (x_event_type or "captured").strip() or "captured",
        "capture_label": "",
        "captured_at": x_captured_at or "",
        "received_at": datetime.now(timezone.utc).isoformat(),
        "blocked_by_vad": _parse_bool(x_blocked_by_vad),
        "max_probability": _parse_float(x_max_probability),
        "average_probability": _parse_float(x_average_probability),
        "probability_cutoff": _parse_int(x_probability_cutoff),
        "peak_probability_cutoff": _parse_int(x_peak_probability_cutoff),
        "active_window_count": _parse_int(x_active_windows),
        "min_active_windows": _parse_int(x_min_active_windows),
        "rise_score": _parse_int(x_rise_score),
        "vad_max_probability": _parse_int(x_vad_max_probability),
        "vad_average_probability": _parse_int(x_vad_average_probability),
        "detection_profile": (x_detection_profile or "").strip(),
        "probability_history": _parse_probability_history(x_probability_history),
        "notes": x_notes or "",
        "converted": result["converted"],
        "detected_format": result["detected_format"],
        "final_format": result["final_format"],
        "postprocess": result["postprocess"],
        "message": result["message"],
        "review_status": "pending",
    }
    _write_sidecar_json(audio_path, sidecar)
    with AUTO_TRAIN_LOCK:
        auto_review_config = dict(AUTO_TRAIN_CONFIG)
    if auto_review_config.get("enabled") and _captured_event_is_auto_reviewable(sidecar, auto_review_config):
        _queue_auto_review(audio_path.name)

    return {
        "ok": True,
        "item": _captured_item_from_path(audio_path),
        "captured_count": len(_list_captured_sample_names()),
    }


@app.get("/api/captured_audio")
def captured_audio():
    takes = _sync_personal_samples_state()
    items = _list_captured_items()
    samples = _samples_payload()
    return {
        "ok": True,
        "items": items,
        "captured_count": len(items),
        "negative_count": samples["negative_count"],
        "personal_count": len(takes),
    }


@app.get("/api/samples")
def samples():
    return _samples_payload()


@app.get("/api/data")
def managed_data():
    return _managed_data_payload()


@app.delete("/api/data/{item_id}")
def delete_managed_data(item_id: str):
    try:
        return _delete_managed_data_item(item_id)
    except KeyError as exc:
        return JSONResponse({"ok": False, "error": str(exc.args[0])}, status_code=404)
    except RuntimeError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=409)
    except OSError as exc:
        return JSONResponse({"ok": False, "error": f"Could not delete trainer data: {exc}"}, status_code=500)


@app.get("/api/audio/{bucket}/{file_name}")
def audio_file(bucket: str, file_name: str):
    bucket_map = {
        "captured": CAPTURED_DIR,
        "personal": PERSONAL_DIR,
        "negative": NEGATIVE_DIR,
    }
    directory = bucket_map.get(bucket)
    if directory is None:
        return JSONResponse({"ok": False, "error": "Unknown audio bucket."}, status_code=404)
    try:
        path = _resolve_audio_path(directory, file_name)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    if bucket == "captured":
        _ensure_captured_playback_ready(path, _load_sidecar_json(path))
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.delete("/api/samples/{bucket}/{file_name}")
def delete_sample(bucket: str, file_name: str):
    bucket_map = {
        "personal": PERSONAL_DIR,
        "negative": NEGATIVE_DIR,
    }
    directory = bucket_map.get(bucket)
    if directory is None:
        return JSONResponse({"ok": False, "error": "Unknown sample bucket."}, status_code=404)
    try:
        path = _resolve_audio_path(directory, file_name)
        metadata = _load_sidecar_json(path)
        _remove_audio_with_sidecar(path)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    if bucket == "negative" and metadata.get("auto_negative"):
        with AUTO_TRAIN_LOCK:
            AUTO_TRAIN_STATE["pending_negative_count"] = max(
                0,
                int(AUTO_TRAIN_STATE.get("pending_negative_count") or 0) - 1,
            )
            _save_auto_train_state_locked()
    return {"ok": True, "deleted_bucket": bucket, "deleted_file": file_name, "message": f"Deleted {file_name}"}


@app.post("/api/samples/{bucket}/{file_name}/vad")
def vad_segments(bucket: str, file_name: str):
    bucket_map = {"personal": PERSONAL_DIR, "negative": NEGATIVE_DIR}
    directory = bucket_map.get(bucket)
    if directory is None:
        return JSONResponse({"ok": False, "error": "Unknown sample bucket."}, status_code=404)
    try:
        path = _resolve_audio_path(directory, file_name)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)

    wav_bytes = path.read_bytes()
    try:
        all_segments = _detect_speech_segments(wav_bytes)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"VAD failed: {str(e)}"}, status_code=500)

    # Only return the first segment longer than 250ms. Add deterministic padding
    # so VAD remains a guide instead of cropping quiet wake-word onsets too tightly.
    filtered = [s for s in all_segments if (s["end"] - s["start"]) >= 0.25]
    if not filtered:
        return {"ok": True, "file_name": file_name, "segments": [], "segment_count": 0}
    seg = filtered[0]
    info = _inspect_wav_bytes(wav_bytes) or {}
    duration_s = float(info.get("duration_s") or 0.0)
    start = max(0.0, round(seg["start"] - VAD_SELECTION_PAD_START_S, 3))
    end = round(seg["end"] + VAD_SELECTION_PAD_END_S, 3)
    if duration_s > 0:
        end = min(duration_s, end)
    if end <= start:
        end = start + 0.001
    segment = {"start": start, "end": end}
    return {"ok": True, "file_name": file_name, "segments": [segment], "segment_count": 1}


@app.post("/api/samples/trim")
async def trim_sample_upload(
    file: UploadFile = File(...),
    bucket: str = Form(...),
    source_file: str = Form(...),
    start_time: str | None = Form(None),
    end_time: str | None = Form(None),
):
    bucket_map = {"personal": PERSONAL_DIR, "negative": NEGATIVE_DIR}
    directory = bucket_map.get(bucket)
    if directory is None:
        return JSONResponse({"ok": False, "error": "Unknown sample bucket."}, status_code=404)

    data = await file.read()
    if not data:
        return JSONResponse({"ok": False, "error": "Empty audio file."}, status_code=400)

    # Validate WAV
    info = _inspect_wav_bytes(data)
    if not info:
        try:
            data = _normalize_audio_to_target_wav(data, file.filename or "trimmed.wav")
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Audio normalization failed: {e}"}, status_code=400)
    elif not _is_target_wav(info):
        try:
            data = _normalize_audio_to_target_wav(data, file.filename or "trimmed.wav")
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Audio normalization failed: {e}"}, status_code=400)

    # Resolve the original file path
    try:
        orig_path = _resolve_audio_path(directory, source_file)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)

    # Backup original to trim_history for undo
    TRIM_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    backup_name = f"{ts}_{source_file}"
    backup_path = TRIM_HISTORY_DIR / backup_name
    shutil.copy2(orig_path, backup_path)
    # Copy sidecar too
    orig_sidecar = _audio_sidecar_path(orig_path)
    if orig_sidecar.exists():
        shutil.copy2(orig_sidecar, _audio_sidecar_path(backup_path))

    # Replace the original file with trimmed audio
    orig_path.write_bytes(data)

    # Update sidecar with trim metadata
    old_sidecar = _load_sidecar_json(orig_path)
    sidecar = {
        **old_sidecar,
        "trimmed": True,
        "source_file": source_file,
        "source_bucket": bucket,
        "trim_start_s": float(start_time) if start_time else None,
        "trim_end_s": float(end_time) if end_time else None,
        "undo_backup_file": backup_name,
    }
    _write_sidecar_json(orig_path, sidecar)

    # Return the updated sample info
    updated_item = _sample_item_from_path(orig_path, bucket)
    updated_item["trimmed"] = True
    updated_item["source_file"] = source_file
    return {"ok": True, "updated_sample": updated_item, "message": f"Trimmed {source_file}"}


@app.post("/api/samples/revert")
def revert_trim(
    bucket: str = Form(...),
    file_name: str = Form(...),
):
    bucket_map = {"personal": PERSONAL_DIR, "negative": NEGATIVE_DIR}
    directory = bucket_map.get(bucket)
    if directory is None:
        return JSONResponse({"ok": False, "error": "Unknown sample bucket."}, status_code=404)

    try:
        file_path = _resolve_audio_path(directory, file_name)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)

    sidecar = _load_sidecar_json(file_path)
    backup_name = sidecar.get("undo_backup_file")
    if not backup_name:
        return JSONResponse({"ok": False, "error": "No trim backup found for this sample."}, status_code=400)

    backup_path = TRIM_HISTORY_DIR / backup_name
    if not backup_path.exists():
        return JSONResponse({"ok": False, "error": "Trim backup file missing."}, status_code=404)

    # Restore original from backup
    shutil.copy2(backup_path, file_path)
    # Restore sidecar
    backup_sidecar = _audio_sidecar_path(backup_path)
    if backup_sidecar.exists():
        shutil.copy2(backup_sidecar, _audio_sidecar_path(file_path))

    # Clean up backup
    backup_path.unlink()
    if backup_sidecar.exists():
        backup_sidecar.unlink()

    updated_item = _sample_item_from_path(file_path, bucket)
    return {"ok": True, "updated_sample": updated_item, "message": f"Reverted {file_name}"}


@app.post("/api/captured_audio/{file_name}/approve_personal")
def approve_captured_audio_to_personal(file_name: str):
    try:
        result = _move_captured_audio(file_name, PERSONAL_DIR, target_prefix="sample", review_status="approved_personal")
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, **result}


@app.post("/api/captured_audio/{file_name}/mark_negative")
def mark_captured_audio_negative(file_name: str):
    try:
        result = _move_captured_audio(file_name, NEGATIVE_DIR, target_prefix="negative", review_status="approved_negative")
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return {"ok": True, **result}


@app.post("/api/captured_audio/{file_name}/discard")
def discard_captured_audio(file_name: str):
    try:
        path = _resolve_audio_path(CAPTURED_DIR, file_name)
        _remove_audio_with_sidecar(path)
    except FileNotFoundError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=404)
    return {"ok": True, "captured_count": len(_list_captured_sample_names())}


@app.get("/api/trained_wake_words/catalog")
def trained_wake_words_catalog(request: Request):
    return {
        "ok": True,
        "base_url": _advertised_base_url(request),
        "wake_words": _list_trained_wake_words(_advertised_base_url(request)),
    }


@app.get("/api/trained_wake_words/{filename}")
def trained_wake_word_artifact(filename: str):
    safe_filename = Path(filename or "").name
    if not safe_filename or Path(safe_filename).suffix.lower() not in {".json", ".tflite"}:
        return JSONResponse({"ok": False, "error": "Unsupported wake word artifact."}, status_code=400)
    _sync_trained_wake_word_artifacts()
    if safe_filename.endswith(ESPHOME_MANIFEST_SUFFIX):
        source_stem = safe_filename[: -len(ESPHOME_MANIFEST_SUFFIX)]
        source_path = TRAINED_WAKE_WORDS_DIR / f"{source_stem}.json"
        if not source_stem or not source_path.is_file():
            return JSONResponse({"ok": False, "error": "Wake word artifact not found."}, status_code=404)
        try:
            metadata = json.loads(source_path.read_text(encoding="utf-8"))
        except Exception:
            return JSONResponse({"ok": False, "error": "Wake word package is invalid."}, status_code=422)
        if not isinstance(metadata, dict):
            return JSONResponse({"ok": False, "error": "Wake word package is invalid."}, status_code=422)
        model_name = Path(str(metadata.get("model") or f"{source_stem}.tflite")).name
        if not (TRAINED_WAKE_WORDS_DIR / model_name).is_file():
            return JSONResponse({"ok": False, "error": "Wake word model not found."}, status_code=404)
        return JSONResponse(
            _esphome_manifest(metadata),
            headers={"Cache-Control": "no-store, max-age=0"},
        )
    artifact_path = TRAINED_WAKE_WORDS_DIR / safe_filename
    if not artifact_path.exists() or not artifact_path.is_file():
        return JSONResponse({"ok": False, "error": "Wake word artifact not found."}, status_code=404)
    media_type = "application/json" if artifact_path.suffix.lower() == ".json" else "application/octet-stream"
    return FileResponse(str(artifact_path), media_type=media_type, filename=artifact_path.name)


@app.post("/api/train")
def train_now(payload: Dict[str, Any] = None):
    payload = payload or {}
    allow_no_personal = bool(payload.get("allow_no_personal", False))
    if TRAINING_SHUTDOWN_EVENT.is_set():
        return JSONResponse(
            {"ok": False, "error": "WakeWord Trainer is shutting down."},
            status_code=503,
        )

    with STATE_LOCK:
        safe_word = STATE["safe_word"]
        language = (STATE.get("language") or DEFAULT_LANGUAGE)
        english_accent = normalize_english_accent(
            STATE.get("english_accent"), language
        )
        tts_mode = normalize_tts_mode(STATE.get("tts_mode"))
        takes_received = int(STATE["takes_received"])
        speakers_total = int(STATE["speakers_total"])
        takes_per_speaker = int(STATE["takes_per_speaker"])
        training_running = bool(STATE["training"]["running"])

    takes_total = speakers_total * takes_per_speaker

    if training_running:
        return JSONResponse({"ok": False, "error": "Training already running"}, status_code=400)

    if not safe_word:
        return JSONResponse({"ok": False, "error": "No active session"}, status_code=400)

    if takes_received == 0 and not allow_no_personal:
        return JSONResponse(
            {
                "ok": False,
                "error": "No personal voice samples uploaded yet.",
                "code": "NO_PERSONAL_SAMPLES",
                "message": "You can train without personal voices, or upload samples first.",
                "takes_total": takes_total,
            },
            status_code=400,
        )

    if not Path(TRAIN_SCRIPT).exists():
        return JSONResponse({"ok": False, "error": f"TRAIN_SCRIPT not found: {TRAIN_SCRIPT}"}, status_code=500)

    with DATA_MANAGEMENT_LOCK:
        with STATE_LOCK:
            if STATE["training"]["running"]:
                return JSONResponse({"ok": False, "error": "Training already running"}, status_code=400)
            try:
                training_lock = _try_acquire_training_run_lock()
            except Exception as exc:
                return JSONResponse(
                    {"ok": False, "error": f"Could not acquire the training lock: {exc}"},
                    status_code=500,
                )
            if training_lock is None:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "Training is already running in another trainer process.",
                        "code": "TRAINING_LOCKED",
                    },
                    status_code=409,
                )
            STATE["training"]["running"] = True
    try:
        _start_training_thread(
            safe_word,
            language,
            False,
            training_lock,
            tts_mode=tts_mode,
            english_accent=english_accent,
        )
    except Exception as exc:
        with STATE_LOCK:
            STATE["training"]["running"] = False
        _release_training_run_lock(training_lock)
        return JSONResponse({"ok": False, "error": f"Could not start training: {exc}"}, status_code=500)

    return {
        "ok": True,
        "started": True,
        "safe_word": safe_word,
        "english_accent": english_accent,
        "tts_mode": tts_mode,
        "personal_samples_used": takes_received > 0,
        "allow_no_personal": allow_no_personal,
    }


@app.get("/api/train_status")
def train_status():
    with STATE_LOCK:
        return {"ok": True, "training": dict(STATE["training"])}


@app.post("/api/reset_recordings")
def reset_recordings():
    _reset_personal_samples_dir()
    takes = _sync_personal_samples_state()
    return {"ok": True, "takes_received": len(takes), "takes": takes}


@app.post("/api/reset_negative_samples")
def reset_negative_samples():
    _reset_audio_dir(NEGATIVE_DIR)
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_STATE["pending_negative_count"] = 0
        _save_auto_train_state_locked()
    return {"ok": True, "negative_count": len(_list_negative_samples())}
