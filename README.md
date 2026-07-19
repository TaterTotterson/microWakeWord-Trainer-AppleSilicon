<div align="center">
  <a href="https://taterassistant.com">
    <img src="images/tater-repo-logo.png" alt="microWakeWord Trainer" width="460"/>
  </a>
</div>
<h3 align="center">
  <a href="https://taterassistant.com">taterassistant.com</a>
</h3>

Train custom microWakeWord models on Apple Silicon with a local web UI, generated Piper samples, device-captured samples, reviewed false-wake negatives, live training logs, and local wake-word links for Tater Native satellites.

Real samples come from device-captured wake audio, close misses, or manual uploads. Every saved sample is normalized to `16 kHz / mono / 16-bit PCM WAV` before training.

---

## What The UI Does

- `Trainer` starts a wake-word session, shows positive/negative sample counts, and launches training.
- `Auto Training` transcribes wake triggers, files phrase-misses as reviewed negatives, retrains on a schedule, and requests a satellite model refresh through Tater.
- `Captured Audio` reviews clips sent by Tater Native or ESPHome sats, including wake hits, close misses, and false wakes.
- `Samples` plays, removes, clears, and manually imports personal or negative samples.
- `Wake Words` lists locally trained JSON/model links for live wake-word switching in Tater.
- Popup consoles show colorized training logs while long-running jobs are active.

---

## macOS App

The easiest way to run the trainer on Apple Silicon is the signed macOS app from the GitHub releases page:

[Download WakeWord Trainer for macOS](https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon/releases/latest)

The app wraps the same local web UI, keeps the capture server running from the menu bar, and opens the trainer in a native macOS window. It stores captured audio, samples, generated models, caches, and local environments in:

```text
~/.taterwakewordtrainer/app/current
```

Use the manual clone flow below if you want to develop the trainer, run directly from source, or inspect the scripts.

---

## Clone The Repo

```bash
git clone https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon.git
cd microWakeWord-Trainer-AppleSilicon
```

---

## Run The Web UI

```bash
./run.sh
```

The launcher:

- requires Python `3.11` by default at `/opt/homebrew/bin/python3.11`
- creates or reuses `.recorder-venv`
- installs the UI dependencies
- serves the app on `0.0.0.0:8789` so satellites can send captured audio

Open:

```text
http://127.0.0.1:8789
```

Useful overrides:

```bash
REC_HOST=127.0.0.1 ./run.sh
REC_PORT=8790 ./run.sh
REC_PYTHON_BIN=/path/to/python3.11 ./run.sh
```

If you change `REC_PORT`, use that same port in the satellite `Trainer App URL`.

---

## macOS Menu Bar App

The native app lives in:

```text
macos/WakeWordTrainer/
```

It wraps the same local web UI, keeps the capture server running from a menu bar item, and opens the trainer in an embedded macOS window. On first launch it copies the bundled trainer source into:

```text
~/.taterwakewordtrainer/app/current
```

Captured audio, samples, generated models, caches, and local virtual environments stay in that support folder so app updates do not wipe training data.

Build locally:

```bash
macos/WakeWordTrainer/scripts/build_app.sh
```

Build the updater zip and manifest:

```bash
macos/WakeWordTrainer/scripts/package_update.sh
```

Build the drag-to-Applications installer DMG:

```bash
macos/WakeWordTrainer/scripts/build_dmg.sh
```

Tagged releases matching the app version, for example `v15`, run `.github/workflows/macos-release.yml`. The workflow builds the updater zip, installer DMG, update manifest, uploads them as workflow/GitHub release assets, and commits the generated release files back to `main`.

Update `WHATS_NEW.md` before creating a release tag. The workflow prepends that curated section to GitHub's automatically generated release notes.

---

## Captured Audio Workflow

To collect samples from a sat, point its trainer feedback setting at this app. Tater Native satellites use the native settings popup in Tater. Older ESPHome satellites can still use their device entities.

For Tater Native satellites, enable trainer feedback in Tater:

- `Send Good Wakes To Trainer` toggles upload of confirmed wake-word triggers.
- `Send Close Misses To Trainer` toggles upload of near misses.
- `Trainer App URL` sets the trainer address, for example `http://trainer.local:8789` or `http://<trainer-ip>:8789`.

For older ESPHome firmware, the equivalent capture setup is exposed as device entities:

- `Capture Wake Audio` toggles upload of wake-word triggers.
- `Capture Close Misses` toggles upload of near misses.
- `Trainer App URL` sets the trainer address, for example `http://<trainer-ip>:8789`.

Satellites send raw captured audio to:

```text
/api/upload_captured_audio_raw
```

Keep the training app running and reachable at the `Trainer App URL` while capture is enabled. The sats upload clips live; if the app is stopped or the URL is wrong, captured audio will not be saved.

In the `Captured Audio` tab:

- play each clip from the inbox
- mark good wake-word clips as `This is good`
- mark bad triggers as `False wake`
- discard clips that should not be used

Approved clips move into:

```text
personal_samples/
```

False wakes move into:

```text
negative_samples/
```

Captured audio is boosted for easier playback in the UI, then kept in the correct training format.

---

## Samples

The `Samples` tab is the sample library.

- `Personal` samples are positive examples of the wake word.
- `Negative` samples are reviewed false wakes or hard negatives.
- Both can be played back and removed one at a time.
- Manual upload is available here as an optional seed path.

Accepted manual upload formats include:

- WAV
- MP3
- M4A
- FLAC
- OGG
- AAC
- OPUS
- WEBM

Uploads are validated or converted with `ffmpeg` into:

```text
16 kHz / mono / 16-bit PCM WAV
```

Starting a new session does not clear samples. Use the clear buttons in `Samples` if you want to remove saved personal or negative clips.

---

## Training Flow

1. Enter the wake phrase in `Trainer`.
2. Choose the language.
3. Optionally test pronunciation with `Test TTS`.
4. Review the positive and negative sample counts.
5. Click `Start training`.
6. Watch the popup training console.

Personal samples are optional. Training can run with zero personal samples after confirmation, using generated TTS samples and the stock negative datasets.

Reviewed negative samples are included as a separate hard-negative feature set when present, so false wakes from your real devices can make the next model more selective.

---

## Auto Training

Auto Training is disabled until it is configured in its own tab.

1. Enter the active wake phrase and STT language.
2. Choose how often training may run and how many new negatives are required.
3. Set the Tater URL (normally `http://127.0.0.1:8501` when Tater runs on the same Mac) and optional API token or satellite selector.
4. Save and enable Auto Training.

New wake-trigger captures are transcribed locally with MLX Whisper. A normal wake trigger moves to `negative_samples/` only when STT returns text and the configured wake phrase is absent. By default, confirmed phrase matches stay in `Captured Audio` for review.

Two optional cleanup rules are available:

- `Delete confirmed good wakes` removes normal wake-trigger clips after STT confirms the configured phrase.
- `Promote confirmed close misses` checks close misses that passed VAD and moves them to the personal positive samples only when STT confirms the configured phrase.

A close miss that was blocked by VAD, has an empty transcript, or does not contain the configured phrase stays in `Captured Audio`; it is never turned into a negative automatically. Captures for another configured wake word also stay out of the automatic path. The transcript and auto-review reason remain in sample or Auto Training state metadata for auditing.

Saving Auto Training settings also scans existing eligible captures. Enabling close-miss promotion reviews previous unreviewed close misses, while enabling cleanup removes previously confirmed good wakes without transcribing them a second time.

The first automatic transcription downloads the configured MLX Whisper model into `auto_train_models/`. Scheduled training only starts after the configured number of new auto-reviewed negatives has accumulated. A successful automatic run asks Tater to re-fetch each connected satellite's current custom wake JSON profile and re-push its native live settings. The current Tater Native firmware treats that settings generation as a forced refresh and downloads the updated model even though its JSON URL has not changed.

Use `Review inbox now`, `Train now`, and `Refresh satellites now` to run each stage manually while testing the setup.

---

## Language Support

The language picker is dynamic.

- `en` is always available.
- English keeps the existing dedicated generator model path.
- Non-English languages are discovered from the Piper voices catalog and any local Piper voice metadata.
- When a non-English language is selected, the trainer downloads all voices for that selected language only.
- Already-downloaded voices are reused.
- It does not download every language up front.

If the upstream Piper catalog is unavailable, already-installed local voices are used when available.

---

## Dataset Behavior

The first training run downloads and prepares the training datasets when they are missing. After the datasets are prepared, later runs reuse the local copies.

Piper voices, generated samples, and feature caches are also reused when the selected language, wake word, and sample inputs have not changed.

---

## Trained Wake Words

The `Wake Words` tab lists locally trained wake-word packages from `trained_wake_words/`.

- Copy the JSON URL into the Tater Native satellite settings to switch wake words live.
- Open the JSON or model links directly for quick inspection.
- The JSON includes the matching model path plus Tater tuning metadata.
- No firmware flashing happens from this trainer app anymore.

Use the main Tater app for satellite firmware updates and USB flashing.

---

## Output Files

Successful runs produce firmware-ready artifacts in:

```text
trained_wake_words/<wake_word>.tflite
trained_wake_words/<wake_word>.json
```

The `Wake Words` tab uses this folder to populate the local wake-word links.

Wake-word links now advertise a LAN-reachable address instead of copying the browser's `127.0.0.1` host. The trainer uses this order:

1. `Trainer public URL` from the Auto Training tab
2. `REC_PUBLIC_BASE_URL`
3. an automatically discovered LAN IPv4 address and `REC_PORT`

Set the public URL explicitly if the Mac has multiple network interfaces or the satellites reach it through a different hostname.

The JSON keeps the standard microWakeWord fields for compatibility:

```json
{
  "micro": {
    "probability_cutoff": 0.97,
    "sliding_window_size": 6
  }
}
```

It also includes Tater Native metadata used by newer satellites and the Tater settings UI:

```json
{
  "model_format": "tflite_stream_state_internal_quant",
  "quantization": "int8",
  "sample_rate": 16000,
  "tater_native": {
    "format_version": 1,
    "wake_threshold": 0.97,
    "wake_sliding_window": 6,
    "close_miss_threshold": 0.80,
    "frontend": {
      "name": "tflm_microfrontend",
      "sample_rate": 16000,
      "feature_duration_ms": 30,
      "feature_step_ms": 10,
      "feature_size": 40
    }
  }
}
```

Calibration metrics are included under `calibration` so false accepts/hour and recall can be surfaced in the UI.
Calibration evaluates thresholds from `0.95` through `1.00` with sliding windows of `5`, `6`, and `7`. Among candidates within 0.5 percentage points of the best recall, it prefers the lowest measured ambient false-accept rate. If calibration cannot complete, packaging uses the conservative `0.97` threshold and a window of `6`.

Intermediate training files are created under:

```text
trained_models/
```

---

## Direct Training Script

Run the Apple Silicon training pipeline directly:

```bash
./train_microwakeword_macos.sh "hey_tater"
```

If `personal_samples/*.wav` or `negative_samples/*.wav` exists, those samples are included automatically.

---

## Important Notes

- Personal samples are optional.
- Negative samples are optional but useful for reducing false wakes.
- The UI server is `trainer_server.py`.
- The launcher is `run.sh`.
- Trainer feedback settings live in Tater for Tater Native satellites, and on device entities for older ESPHome satellites.

---

## Credits

Built on top of:

- [microWakeWord](https://github.com/kahrendt/microWakeWord)
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
