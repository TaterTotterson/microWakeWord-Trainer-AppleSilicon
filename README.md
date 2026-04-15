<div align="center">
  <h1>microWakeWord Apple Silicon Trainer UI</h1>
  <img width="800" alt="Screenshot 2026-04-14 at 11 02 06 PM" src="https://github.com/user-attachments/assets/477fb140-fb7f-4ca7-9a03-3db938c5a826" />
</div>

Train custom microWakeWord models on Apple Silicon with:

- uploaded personal voice samples
- automatically generated Piper TTS samples
- a local web UI with live training logs

The project no longer records audio in the browser. Instead, you upload your own audio files, the app validates or converts them into the required training format, and training runs from the same UI.

---

## What The UI Does

- Start a session for a wake word
- Test TTS pronunciation
- Upload one or many personal voice samples
- Automatically normalize uploads to `16 kHz / mono / 16-bit PCM WAV`
- Train with or without personal samples
- Show training output in a popup console window

Personal samples are optional. If you upload none, the trainer can still run with TTS-only data after confirmation.

---

## Clone The Repo

```bash
git clone https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon.git
cd microWakeWord-Trainer-AppleSilicon
```

---

## Run The Web UI

```bash
./run_recorder_macos.sh
```

What this does:

- creates or reuses `.recorder-venv`
- installs the UI server dependencies once
- launches the FastAPI app locally

Then open:

```text
http://127.0.0.1:8789
```

---

## Personal Samples

The UI accepts common audio formats such as:

- WAV
- MP3
- M4A
- FLAC
- OGG
- AAC
- OPUS
- WEBM

Uploads are checked and, if needed, converted with `ffmpeg` into:

```text
16 kHz / mono / 16-bit PCM WAV
```

Converted files are stored in:

```text
personal_samples/
```

Notes:

- samples are not auto-cleared when you start a new session
- use the `Clear personal samples` button if you want to remove them
- training can run with zero personal samples

---

## Language Support

The language picker is dynamic.

- `en` is always available
- non-English languages are populated from Piper voice metadata
- when you start training with a non-English language, the trainer downloads all Piper ONNX voices for that selected language only
- it does not download every language up front
- already-downloaded voices are reused

English stays on its existing dedicated generator model path. Non-English languages use Piper ONNX voices for the selected language family.

If the upstream Piper catalog is unavailable, already-installed local voices can still be used.

---

## Training Flow

1. Enter the wake word
2. Optionally test pronunciation with `Test TTS`
3. Optionally upload personal samples
4. Click `Start training`
5. Watch the popup console for:
   - selected-language voice downloads when needed
   - sample generation progress
   - training progress and final status

The console can be reopened with `Open console`.

---

## Training Script Only

You can still run the Apple Silicon training pipeline directly:

```bash
./train_microwakeword_macos.sh "hey_tater"
```

If `personal_samples/*.wav` exists, those files are included automatically.

---

## Output Files

Successful runs produce normalized output files such as:

```text
output/<wake_word>.tflite
output/<wake_word>.json
```

If a file with the same name already exists, the trainer keeps it by creating timestamped backup files first.

---

## Notes

- browser microphone recording has been removed from this project
- personal samples are optional, not required
- the UI server module is now `trainer_server.py`
- the launcher script name is still `run_recorder_macos.sh` for compatibility

---

## Credits

Built on top of:

- [microWakeWord](https://github.com/kahrendt/microWakeWord)
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
