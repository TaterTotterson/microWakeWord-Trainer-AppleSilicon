# Advanced trainer guide

[Back to the README](../README.md)

Detailed recording, training, model-file, and development reference. Run shell
commands from the repository root unless a section says otherwise.

## Run from source

### Clone the repo

```bash
git clone https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon.git
cd microWakeWord-Trainer-AppleSilicon
```

### Start the web UI

```bash
./run.sh
```

The launcher:

- requires Python `3.11` by default at `/opt/homebrew/bin/python3.11`
- creates or reuses `~/.taterwakewordtrainer/recorder-venv`
- installs the UI dependencies
- serves the app on `0.0.0.0:8789` so satellites can send captured audio

Running from a Git clone still keeps datasets, samples, generated features, models, logs, caches, and local environments under `~/.taterwakewordtrainer`. The source checkout is not used as a training-data directory.

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

## Captured Audio Workflow

To collect samples from a sat, point its trainer feedback setting at this app. Tater Native satellites use their settings in Tater or the Home Assistant bridge. Audio uploads do not require the optional auto-training link. Older ESPHome satellites can still use their device entities.

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

In **Captures**:

- play each clip from the inbox
- choose **Add positive** for good wake-word clips
- choose **Mark negative** for false triggers
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

## Training Flow

1. On `Start`, choose `Create a wake word`, or open a saved word to review results and JSON links without starting a session.
2. For a new word, enter the phrase, language, English accent, and voice-generation source. `System preview` checks pronunciation with the system voice.
3. Add or review personal recordings and negatives. Upload, playback, trimming, VAD, restore, and bulk import remain available in the sample manager.
4. Select any combination of `MicroWakeWord`, `openWakeWord`, `NanoWakeWord · CNN`, and `NanoWakeWord · GRU`, plus optional phrases to reject. Recovery actions are under `Additional training actions`.
5. Review the configuration and click `Start training`. The `Training` page shows the current stage and logs; training continues while you browse.
6. Open the completed word for its measured validation results and JSON links. For an existing phrase, `Train a new version` reuses its saved data.

Personal samples are optional. Training can run with zero personal samples after confirmation, using generated TTS samples and the stock negative datasets.

### Saved words, cached datasets, and model history

Every phrase started in the Trainer is added to the **Saved wake words** library. Switching to another saved word restores that word's personal samples, reviewed negatives, generated TTS corpus, augmented features, boundary examples, and retained training workspace. The previous word is archived under `wake_word_library/words/<safe_word>/assets/` in the selected data folder, so changing the active phrase no longer discards or overwrites its expensive data.

Each completed run also creates an immutable snapshot under `wake_word_library/words/<safe_word>/runs/`. The **Wake Words** tab shows the run history, selected engines, recall/false-trigger metrics, thresholds, and quality rating so multiple attempts can be compared and open any saved version for its own bundle JSON and ESPHome JSON URLs. These version-specific links keep referencing that version’s model bytes. `Use v… as current` validates the saved artifacts and changes the current publication without deleting recordings, caches, or other versions. The stable published JSON URL follows the selected current version and remains on the committed snapshot while another run trains. Failed attempts remain in history and cannot replace the current publication. Publish through Tater separately when satellites should switch to the selected version.

After a word has at least one detector bundle, **Retrain boundary only** rebuilds and calibrates only the wake-end model. It preserves MWW, OWW, and NWW detector weights and reuses the saved word's cached corpus and features. The direct script accepts `--boundary-only`, or `MWW_BOUNDARY_ONLY=1` for app/server launches:

```bash
./train_microwakeword_macos.sh "hey tater" 50000 100 --boundary-only
```

Boundary-only mode requires the retained detector training config from an earlier full MicroWakeWord run.

Reviewed negative samples are included as a separate hard-negative feature set when present, so false wakes from your real devices can make the next model more selective.

The first OWW or NWW run downloads its separate trainer stack and caches it for later runs. NanoWakeWord reuses this trainer's direct-generated positive TTS corpus, then adds personal positives, reviewed negatives, and NanoWakeWord's own adversarial negatives and held-out validation clips. CNN is the lightweight default architecture; GRU may be selected by itself or alongside CNN. The streaming gate automatically calibrates threshold and consecutive-hit patience, and the validated policy is stored with each model in the bundle. Each successful stage updates only its section of the canonical Tater bundle, so mix-and-match runs preserve other valid models already built for the phrase while removing obsolete NWW architectures.

## Auto Training

Auto Training is disabled until it is configured in its own tab.

1. Enter the active wake phrase, STT language, and local STT engine.
2. Choose how often training may run and how many new negatives are required.
3. Optionally, for automatic model publishing, set the Tater or Home Assistant bridge URL (normally `http://127.0.0.1:8501` when Tater runs on the same Mac), click `Link Tater`, and enter the short-lived code shown in Tater Voice Settings. After training, the trainer tells Tater which new wake-word JSON is active and Tater broadcasts it to every satellite.
4. Save and enable Auto Training.

New wake-trigger captures are transcribed locally with the engine selected in Auto Training. Faster Whisper is the recommended default, Parakeet ONNX provides a multilingual ONNX option, and MLX Whisper remains available for users who prefer it. A normal wake trigger moves to `negative_samples/` only when STT returns text and the configured wake phrase is absent. By default, confirmed phrase matches stay in `Captured Audio` for review.

Two optional cleanup rules are available:

- `Delete confirmed good wakes` removes normal wake-trigger clips after STT confirms the configured phrase.
- `Promote confirmed close misses` checks close misses that passed VAD and moves them to the personal positive samples only when STT confirms the configured phrase.

A close miss that was blocked by VAD, has an empty transcript, or does not contain the configured phrase stays in `Captured Audio`; it is never turned into a negative automatically. Captures for another configured wake word also stay out of the automatic path. The transcript and auto-review reason remain in sample or Auto Training state metadata for auditing.

Saving Auto Training settings also scans existing eligible captures. Enabling close-miss promotion reviews previous unreviewed close misses, while enabling cleanup removes previously confirmed good wakes without transcribing them a second time.

The UI intentionally exposes only the STT engine. The trainer manages the matching model and runtime settings: Faster Whisper uses `small.en` for English and `small` for other languages, MLX Whisper uses the matching managed base model, and Parakeet uses the INT8 `nemo-parakeet-tdt-0.6b-v3` model. Models download on first use into `auto_train_models/` and are reused. Scheduled training only starts after the configured number of new auto-reviewed negatives has accumulated. A successful automatic run securely publishes the trained wake-word name and JSON URL to the linked Tater instance. Tater saves it as the global satellite wake word and pushes the updated setting to every connected satellite.

Use `Review inbox now`, `Train now`, and `Publish current wake word now` to run each stage manually while testing the setup.

The auto-training link is optional. Local review and training can be used without pairing; disable automatic activation if you are not linking a controller. For Home Assistant pairing and shared wake-profile behavior, see the [bridge guide](https://github.com/TaterTotterson/Tater-Home-Assistant-Satellites/blob/main/docs/user-guide.md#link-the-wake-word-trainer).

## Language Support

The language picker is built from the live OmniVoice catalog and the installed modern/legacy engines.

- OmniVoice supplies the broad multilingual route (hundreds of catalog languages).
- Qwen3-TTS joins the ensemble for Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian.
- MOSS-TTS-Nano joins the ensemble for its published multilingual set.
- Languages covered by all three engines are marked `Recommended`; smaller ensembles are marked `Supported` or `Experimental`.
- Qwen, OmniVoice, and Piper generate final corpus takes directly. MOSS Nano is clone-only, so each MOSS take uses a different already-accepted direct take as its carrier instead of cycling a small profile bank.
- Qwen exposes 18,750 balanced voice conditions before any instruction repeats, while fresh sampling seeds add further variation. Piper uses the full speaker set in its installed model.
- Every candidate is checked for static, broadband/high-frequency noise, silence, clipping, excessive duration/rambling, and exact duplication before it can enter training. Failed provider shares are filled by a safer provider.

Model environments and weights download on first use and are cached under the selected data folder (by default `~/.taterwakewordtrainer/app/current`). The Qwen and MOSS paths use MLX-Audio on Apple Silicon; OmniVoice uses PyTorch MPS. These environments are isolated from the TensorFlow training environment.

`Four-provider ensemble` uses OmniVoice, Qwen, MOSS, and Piper when a compatible Piper model exists, and safely falls back to the modern providers where it does not. `Modern only` excludes Piper, and `Piper only` preserves the legacy comparison route.

## Dataset Behavior

The first training run downloads and prepares the training datasets when they are missing. After the datasets are prepared, later runs reuse the local copies.

Model downloads, completed generated corpora, and feature caches are reused when the selected language, wake word, TTS mode, and sample inputs have not changed. Updating the macOS app does not invalidate a matching TTS corpus; only a meaningful generation input change or an incomplete corpus rebuilds it.

## Output Files

Successful runs produce the selected firmware-ready artifacts in:

```text
trained_wake_words/<wake_word>.tflite
trained_wake_words/<wake_word>.oww.onnx         # openWakeWord classifier
trained_wake_words/<wake_word>.oww.verifier.pkl # when positive + negative clips exist
trained_wake_words/<wake_word>.nww.cnn.onnx     # NanoWakeWord CNN classifier
trained_wake_words/<wake_word>.nww.gru.onnx     # NanoWakeWord GRU classifier
trained_wake_words/<wake_word>.boundary.tflite  # detector-independent wake-end model
trained_wake_words/<wake_word>.wake-bundle.json
trained_wake_words/<wake_word>.esphome.json     # MWW-only ESPHome manifest
```

Tater always receives `<wake_word>.wake-bundle.json`. Schema v4 embeds the complete metadata for every model that has been built, including hashes, thresholds, validation results, available NWW architectures, and one detector-independent `wake_boundary` TFLite artifact used to locate the acoustic end of the phrase. It does not point at a second MWW metadata JSON. ESPHome remains deliberately separate and MWW-only through `<wake_word>.esphome.json`.

The boundary artifact is trained and threshold-calibrated separately for each wake word. Wake-only clips remain positive while deterministic wake-plus-immediate-speech clips are hard negatives, forcing a causal score peak at the phrase endpoint. Packaging fails if that dedicated model was not exported; the normal wake detector is never substituted as boundary weights.

Wake-word links now advertise a LAN-reachable address instead of copying the browser's `127.0.0.1` host. The trainer uses this order:

1. `Trainer public URL` from the Auto Training tab
2. `REC_PUBLIC_BASE_URL`
3. an automatically discovered LAN IPv4 address and `REC_PORT`

Set the public URL explicitly if the Mac has multiple network interfaces or the satellites reach it through a different hostname.

When MWW is selected, the bundle's `micro_wake_word` section keeps the standard microWakeWord fields:

```json
{
  "micro_wake_word": {
    "micro": {
      "probability_cutoff": 0.97,
      "sliding_window_size": 6
    }
  }
}
```

That same embedded section includes Tater Native metadata used by newer satellites and the Tater settings UI:

```json
{
  "micro_wake_word": {
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
}
```

Calibration metrics are included under `calibration` so false accepts/hour and recall can be surfaced in the UI.
Calibration evaluates thresholds from `0.95` through `1.00` with sliding windows of `5`, `6`, and `7`. Among candidates within 0.5 percentage points of the best recall, it prefers the lowest measured ambient false-accept rate. If calibration cannot complete, packaging uses the conservative `0.97` threshold and a window of `6`.

Intermediate training files are created under:

```text
trained_models/
```

## Direct Training Script

Run the Apple Silicon training pipeline directly:

```bash
./train_microwakeword_macos.sh "hey_tater"
```

The direct script defaults to the four-provider route and generates every final corpus take directly. Compatibility examples:

```bash
./train_microwakeword_macos.sh "hey_tater" 50000 8 --language en --tts-mode hybrid
./train_microwakeword_macos.sh "hey_tater" 50000 8 --language en --tts-mode modern
./train_microwakeword_macos.sh "hey_tater" 50000 100 --language en --tts-mode piper
```

If `personal_samples/*.wav` or `negative_samples/*.wav` exists, those samples are included automatically.

## macOS app builds and releases

The app can move training data to an external drive from **Data & storage**
or the menu bar. Choose an APFS or other Mac-compatible writable volume and
keep it connected while the trainer is running. The app creates a **WakeWord
Trainer Data** folder, copies and verifies the data, then switches the saved
location and restarts. The small launcher runtime, recovery logs, and managed
base Python remain under `~/.taterwakewordtrainer/` so the app can report a
missing or disconnected data drive.

The native app lives in:

```text
macos/WakeWordTrainer/
```

It wraps the same local web UI, keeps the capture server running from a menu bar item, and opens the trainer in an embedded macOS window. On first launch it copies the bundled trainer source into:

```text
~/.taterwakewordtrainer/app/current
```

Captured audio, samples, generated models, caches, and local training environments use the selected data folder and are kept across app updates. The default remains `~/.taterwakewordtrainer/app/current` until it is changed from the Data tab or menu bar.

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

Tagged releases matching the app version, for example `v47.10`, run `.github/workflows/macos-release.yml`. The workflow builds the updater zip, installer DMG, update manifest, uploads them as workflow/GitHub release assets, and commits the generated release files back to `main`.

Update `WHATS_NEW.md` before creating a release tag. The workflow prepends that curated section to GitHub's automatically generated release notes.

## UI development

The production bundle is committed under `static/ui`, so users do not need Node.js. Edit `frontend/src`, then build from the repository root:

```bash
cd frontend
npm install
npm run build
npm test
```

The build type-checks the Vue components and writes the offline bundle used by `trainer_server.py`. Tests require Node.js 22.6 or newer.

The UI uses `/api/events` (server-sent events), reconnects automatically, and falls back to polling. Returning to a browser tab catches up without resetting unsaved edits. Background library updates avoid recursive storage scans. If using a reverse proxy, allow unbuffered streaming responses on `/api/events`.

The UI server is `trainer_server.py`; its launcher is `run.sh`. Both launchers bound the shutdown wait for open live connections.

## Credits

Built on top of:

- [microWakeWord](https://github.com/kahrendt/microWakeWord)
- [OmniVoice](https://github.com/k2-fsa/OmniVoice)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- [MLX-Audio](https://github.com/Blaizzy/mlx-audio)
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) (optional legacy mode)
