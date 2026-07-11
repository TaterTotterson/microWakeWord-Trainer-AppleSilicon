<div align="center">
  <a href="https://taterassistant.com">
    <img src="images/tater-repo-logo.png" alt="microWakeWord Trainer" width="460"/>
  </a>
</div>
<h3 align="center">
  <a href="https://taterassistant.com">taterassistant.com</a>
</h3>

Train custom microWakeWord models on Apple Silicon with a local web UI, generated Piper samples, device-captured samples, reviewed false-wake negatives, live training logs, and prebuilt Tater firmware flashing.

Real samples come from device-captured wake audio, close misses, or manual uploads. Every saved sample is normalized to `16 kHz / mono / 16-bit PCM WAV` before training.

---

## What The UI Does

- `Trainer` starts a wake-word session, shows positive/negative sample counts, and launches training.
- `Captured Audio` reviews clips sent by Tater Native or ESPHome sats, including wake hits, close misses, and false wakes.
- `Samples` plays, removes, clears, and manually imports personal or negative samples.
- `Firmware` pulls verified prebuilt Tater firmware images from GitHub and flashes supported satellites over OTA.
- Popup consoles show colorized training and firmware logs while long-running jobs are active.

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
- installs the UI and firmware flasher dependencies
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

Tagged releases matching the app version, for example `v7`, run `.github/workflows/macos-release.yml`. The workflow builds the updater zip, installer DMG, update manifest, uploads them as workflow/GitHub release assets, and commits the generated release files back to `main`.

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

## Firmware Flashing

The `Firmware` tab flashes prebuilt Tater firmware for supported satellites.

- Downloads the latest prebuilt firmware manifest plus OTA and USB factory images from [`TaterTotterson/Tater-Native-Firmware`](https://github.com/TaterTotterson/Tater-Native-Firmware).
- Verifies downloaded images by size and SHA before upload.
- Auto-detects compatible devices with mDNS when available.
- Allows manual IP or hostname entry if discovery does not find the device.
- Saves the selected OTA target for each firmware family.
- Flashes the prebuilt factory image over Browser USB for first installs or recovery when opened in Chrome or Edge.
- Lists locally trained wake words from `trained_wake_words/` for live model switching.
- Streams download, verification, and OTA upload progress in a colorized firmware console.

> **Tater only:** these native firmware images connect to Tater. They are not Home Assistant or ESPHome satellite firmware.

You usually only flash for firmware updates. New satellites, or devices not already running Tater Native Firmware `v1`, need one USB flash first before OTA updates and live wake-word switching are available.

---

## Output Files

Successful runs produce firmware-ready artifacts in:

```text
trained_wake_words/<wake_word>.tflite
trained_wake_words/<wake_word>.json
```

The firmware tab uses this folder to populate the wake-word dropdown.

The JSON keeps the standard microWakeWord fields for compatibility:

```json
{
  "micro": {
    "probability_cutoff": 0.97,
    "sliding_window_size": 5
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
    "wake_sliding_window": 5,
    "close_miss_threshold": 0.78,
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
- Firmware capture settings live in Tater for Tater Native satellites, and on device entities for older ESPHome satellites.

---

## Credits

Built on top of:

- [microWakeWord](https://github.com/kahrendt/microWakeWord)
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
