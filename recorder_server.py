# recorder_server.py
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
PERSONAL_DIR = ROOT_DIR / "personal_samples"
TRAIN_SCRIPT = os.environ.get("TRAIN_SCRIPT", str(ROOT_DIR / "train_microwakeword_macos.sh"))
DEFAULT_LANGUAGE = os.environ.get("MWW_LANGUAGE", "en")

TAKES_PER_SPEAKER_DEFAULT = int(os.environ.get("REC_TAKES_PER_SPEAKER", "10"))
SPEAKERS_TOTAL_DEFAULT = int(os.environ.get("REC_SPEAKERS_TOTAL", "1"))

app = FastAPI(title="microWakeWord Local Recorder")

# Serve static UI
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def safe_name(raw: str) -> str:
    s = (raw or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    s = re.sub(r"^_+|_+$", "", s)
    return s or "wakeword"


# -------------------- In-memory session state --------------------
STATE: Dict[str, Any] = {
    "raw_phrase": None,
    "safe_word": None,
    "language": DEFAULT_LANGUAGE,

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


def _reset_personal_samples_dir():
    PERSONAL_DIR.mkdir(parents=True, exist_ok=True)
    for p in PERSONAL_DIR.glob("*.wav"):
        try:
            p.unlink()
        except Exception:
            pass


def _append_train_log(line: str):
    line = (line or "").rstrip("\n")
    with STATE_LOCK:
        buf: List[str] = STATE["training"]["log_lines"]
        buf.append(line)
        if len(buf) > 250:
            del buf[: len(buf) - 250]


def _run_training_background(safe_word: str, language: str):
    language = (language or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE
    cmd = ["bash", TRAIN_SCRIPT, safe_word]

    with STATE_LOCK:
        STATE["training"]["running"] = True
        STATE["training"]["exit_code"] = None
        STATE["training"]["log_lines"] = []
        STATE["training"]["safe_word"] = safe_word
        log_path = str(ROOT_DIR / "recorder_training.log")
        STATE["training"]["log_path"] = log_path

    _append_train_log(f"→ Running: {' '.join(cmd)}")
    _append_train_log(f"→ Language: {language}")

    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            env = os.environ.copy()
            env["MWW_LANGUAGE"] = language
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                _append_train_log(line)

            rc = proc.wait()

        _append_train_log(f"✓ Training finished (exit_code={rc})")
        with STATE_LOCK:
            STATE["training"]["exit_code"] = rc

    except Exception as e:
        _append_train_log(f"✗ Training crashed: {e!r}")
        with STATE_LOCK:
            STATE["training"]["exit_code"] = 999

    finally:
        with STATE_LOCK:
            STATE["training"]["running"] = False


# -------------------- Routes --------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h3>Missing UI</h3><p>Create <code>static/index.html</code>.</p>",
            status_code=500,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/api/start_session")
def start_session(payload: Dict[str, Any]):
    raw = (payload.get("phrase") or "").strip()
    if not raw:
        return JSONResponse({"ok": False, "error": "phrase is required"}, status_code=400)

    safe = safe_name(raw)

    speakers_total = int(payload.get("speakers_total") or SPEAKERS_TOTAL_DEFAULT)
    takes_per_speaker = int(payload.get("takes_per_speaker") or TAKES_PER_SPEAKER_DEFAULT)
    language = (payload.get("language") or DEFAULT_LANGUAGE).strip().lower() or DEFAULT_LANGUAGE

    speakers_total = max(1, min(10, speakers_total))
    takes_per_speaker = max(1, min(50, takes_per_speaker))

    with STATE_LOCK:
        STATE["raw_phrase"] = raw
        STATE["safe_word"] = safe
        STATE["language"] = language
        STATE["speakers_total"] = speakers_total
        STATE["takes_per_speaker"] = takes_per_speaker
        STATE["takes_received"] = 0
        STATE["takes"] = []
        # do not interrupt training if running

    # wipe recordings whenever a new session starts
    _reset_personal_samples_dir()

    return {
        "ok": True,
        "raw_phrase": raw,
        "safe_word": safe,
        "language": language,
        "speakers_total": speakers_total,
        "takes_per_speaker": takes_per_speaker,
        "takes_total": speakers_total * takes_per_speaker,
    }


@app.get("/api/session")
def get_session():
    with STATE_LOCK:
        return {
            "ok": True,
            "raw_phrase": STATE["raw_phrase"],
            "safe_word": STATE["safe_word"],
            "language": STATE["language"],
            "speakers_total": STATE["speakers_total"],
            "takes_per_speaker": STATE["takes_per_speaker"],
            "takes_received": STATE["takes_received"],
            "takes": list(STATE["takes"]),
            "training": dict(STATE["training"]),
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

    PERSONAL_DIR.mkdir(parents=True, exist_ok=True)

    # must be this naming format for training script pickup
    out_name = f"speaker{speaker_index:02d}_take{take_index:02d}.wav"
    out_path = PERSONAL_DIR / out_name

    data = await file.read()
    if not data or len(data) < 44:
        return JSONResponse({"ok": False, "error": "Empty/invalid file"}, status_code=400)

    out_path.write_bytes(data)

    with STATE_LOCK:
        if out_name not in STATE["takes"]:
            STATE["takes"].append(out_name)
            STATE["takes_received"] = len(STATE["takes"])

    return {"ok": True, "saved_as": out_name, "takes_received": STATE["takes_received"]}


@app.post("/api/train")
def train_now(payload: Dict[str, Any] = None):
    payload = payload or {}
    allow_no_personal = bool(payload.get("allow_no_personal", False))

    with STATE_LOCK:
        safe_word = STATE["safe_word"]
        language = (STATE.get("language") or DEFAULT_LANGUAGE)
        takes_received = int(STATE["takes_received"])
        speakers_total = int(STATE["speakers_total"])
        takes_per_speaker = int(STATE["takes_per_speaker"])
        training_running = bool(STATE["training"]["running"])

    takes_total = speakers_total * takes_per_speaker

    if training_running:
        return JSONResponse({"ok": False, "error": "Training already running"}, status_code=400)

    if not safe_word:
        return JSONResponse({"ok": False, "error": "No active session"}, status_code=400)

    min_required = max(1, min(3, takes_total))

    if takes_received == 0 and not allow_no_personal:
        return JSONResponse(
            {
                "ok": False,
                "error": f"No personal voice samples recorded (0/{takes_total}).",
                "code": "NO_PERSONAL_SAMPLES",
                "message": "You can train without personal voices, or record samples first.",
                "takes_total": takes_total,
            },
            status_code=400,
        )

    if 0 < takes_received < min_required:
        return JSONResponse(
            {
                "ok": False,
                "error": f"Not enough takes yet ({takes_received}/{takes_total}).",
                "code": "NOT_ENOUGH_TAKES",
                "min_required": min_required,
                "takes_total": takes_total,
            },
            status_code=400,
        )

    if not Path(TRAIN_SCRIPT).exists():
        return JSONResponse({"ok": False, "error": f"TRAIN_SCRIPT not found: {TRAIN_SCRIPT}"}, status_code=500)

    t = threading.Thread(target=_run_training_background, args=(safe_word, language), daemon=True)
    t.start()

    return {
        "ok": True,
        "started": True,
        "safe_word": safe_word,
        "personal_samples_used": takes_received >= min_required,
        "allow_no_personal": allow_no_personal,
    }


@app.get("/api/train_status")
def train_status():
    with STATE_LOCK:
        return {"ok": True, "training": dict(STATE["training"])}


@app.post("/api/reset_recordings")
def reset_recordings():
    _reset_personal_samples_dir()
    with STATE_LOCK:
        STATE["takes_received"] = 0
        STATE["takes"] = []
    return {"ok": True}
