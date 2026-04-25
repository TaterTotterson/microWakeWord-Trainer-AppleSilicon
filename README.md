<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/27661678-cab6-455d-b31c-41383bd0a3e3" 
    width="300"
  />
</p>
<h3 align="center">
  <a href="https://taterassistant.com">taterassistant.com</a>
</h3>

Train custom microWakeWord models on Apple Silicon with a local web UI, generated Piper samples, device-captured samples, reviewed false-wake negatives, live training logs, and ESPHome firmware flashing.

Real samples come from device-captured wake audio, close misses, or manual uploads. Every saved sample is normalized to `16 kHz / mono / 16-bit PCM WAV` before training.

---

## What The UI Does

- `Trainer` starts a wake-word session, shows positive/negative sample counts, and launches training.
- `Captured Audio` reviews clips sent by ESPHome sats, including wake hits, close misses, and false wakes.
- `Samples` plays, removes, clears, and manually imports personal or negative samples.
- `Firmware` builds the latest `microWakeWords` ESPHome YAMLs from GitHub and flashes VoicePE or Satellite1 over OTA.
- Popup consoles show colorized training and firmware logs while long-running jobs are active.

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
- installs the UI, ESPHome, and firmware flasher dependencies
- serves the app on `0.0.0.0:8789` so ESPHome devices can send captured audio

Open:

```text
http://127.0.0.1:8789
```

Useful overrides:

```bash
REC_HOST=127.0.0.1 ./run.sh
REC_PORT=8790 ./run.sh
REC_PYTHON_BIN=/path/to/python3.11 ./run.sh
REC_ESPHOME_VERSION=2026.4.0 ./run.sh
```

If you change `REC_PORT`, use that same port in the ESPHome `Trainer App URL`.

---

## Captured Audio Workflow

To collect samples from a sat, flash it with the Tater firmware from [TaterTotterson/microWakeWords](https://github.com/TaterTotterson/microWakeWords). The `Firmware` tab can build and flash the VoicePE or Satellite1 YAMLs directly from that repo.

After flashing, the device exposes ESPHome entities for capture setup:

- `Capture Wake Audio` toggles upload of wake-word triggers.
- `Capture Close Misses` toggles upload of near misses.
- `Trainer App URL` sets the trainer address, for example `http://<trainer-ip>:8789`.

ESPHome devices can send raw captured audio to:

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

The `Firmware` tab builds and flashes Tater firmware for supported ESPHome sats.

- Downloads the latest firmware YAML templates from `TaterTotterson/microWakeWords` on GitHub.
- Lets you choose `VoicePE` or `Satellite1`.
- Auto-detects ESPHome devices with mDNS when available.
- Allows manual IP or hostname entry if discovery does not find the device.
- Saves firmware form values so you do not re-enter sounds and URLs every run.
- Lists locally trained wake words from `trained_wake_words/` for easy model selection.
- Builds with ESPHome and flashes OTA.
- Streams ESPHome output in a colorized firmware console.

Firmware YAMLs are intentionally pulled from GitHub each time. There is no local fallback path in the trainer UI.

---

## Output Files

Successful runs produce firmware-ready artifacts in:

```text
trained_wake_words/<wake_word>.tflite
trained_wake_words/<wake_word>.json
```

The firmware tab uses this folder to populate the wake-word dropdown.

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
- Firmware capture settings live on the ESPHome device and can be toggled from the device entities after flashing.

---

## Credits

Built on top of:

- [microWakeWord](https://github.com/kahrendt/microWakeWord)
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator)
