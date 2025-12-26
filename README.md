<div align="center">
  <img src="https://raw.githubusercontent.com/TaterTotterson/microWakeWord-Trainer-Docker/refs/heads/main/mmw.png" alt="MicroWakeWord Trainer Logo" width="100" />
  <h1>microWakeWord Trainer Apple Silicon</h1>
</div>

> **Note:** The script will automatically install **ffmpeg** and **wget** via Homebrew
if they are missing. Homebrew itself must already be installed:
https://brew.sh/

## **Quick Start:**
Clone the repo and enter the folder:
```bash
git clone https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon.git
cd microWakeWord-Trainer-AppleSilicon
```
### Train with the default settings (wake word = "hey_tater")
```bash
./train_microwakeword_macos.sh "hey_tater"
```
This will train a model for `hey_tater` and produce `hey_tater.tflite` + `hey_tater.json`
ready for ESPHome integration.

---

### Command Usage
```bash
./train_microwakeword_macos.sh [WAKE_WORD] [MAX_TTS_SAMPLES] [BATCH_SIZE] [--piper-model path.pt] [...]
```
| Argument              | Default     | Description |
|----------------------|-------------|-------------|
| WAKE_WORD            | hey_tater  | The phrase to train on. Will be used for file naming (`<wake_word>.tflite`, `<wake_word>.json`). |
| MAX_TTS_SAMPLES      | 50000       | Number of synthetic TTS samples to generate (skipped if already present). |
| BATCH_SIZE           | 100         | TTS batch size for Piper sample generation. Higher = faster, but uses more memory. |
| --piper-model PATH   | *(optional)*| Path to a custom Piper voice model (.pt or .onnx). Can be repeated for multi-speaker datasets. |

Examples:

### Train with a custom wake word (uses defaults for TTS sample count + batch size)
```bash
./train_microwakeword_macos.sh "hey_tater"
```
### Train with fewer samples (faster)
```bash
./train_microwakeword_macos.sh "ok_mango" 20000
```
### Train with a custom batch size (controls Piper generation speed/memory usage)
```bash
./train_microwakeword_macos.sh "hey_robot" 50000 256
```
### Train with a Custom Piper Voice

You can specify a Piper voice to generate samples for your wake word.  
For Apple Silicon, it’s recommended to use the **multi-speaker** model:

./train_microwakeword_macos.sh "hey_phooey" 50000 100 \
  --piper-model piper-sample-generator/models/en_US-libritts_r-medium.pt

## 🎤 Optional: Personal Voice Samples (Highly Recommended)

In addition to synthetic TTS samples, the trainer can optionally use your own real voice recordings to significantly improve accuracy for your voice and environment.

### How it works
- If a folder named personal_samples/ exists and contains .wav files, the trainer will:
  - Automatically extract features from those recordings
  - Include them during training alongside the synthetic TTS data
  - Up-weight your personal samples during training for better real-world performance

No extra flags or configuration are required — it is detected automatically.

### How to use it
1. Create a folder in the repo root:
   mkdir personal_samples

2. Record yourself saying the wake word naturally and save the files as .wav:
   personal_samples/
     hey_tater_01.wav
     hey_tater_02.wav
     hey_tater_03.wav
     ...

3. Run the training script as normal:
   ./train_microwakeword_macos.sh "hey_tater"

If personal samples are found, you’ll see a message during training indicating they are being included.

### Recording tips
- 10–30 recordings is usually enough to see a noticeable improvement
- Vary distance, volume, and tone slightly
- Record in the same environment where the wake word will be used (room noise matters)
- Use 16-bit WAV files if possible (most recorders do this by default)

Notes:
- `en_US-libritts_r-medium.pt` is a multi-speaker model and will automatically generate varied voices, making your model more robust.
- `.pt` voices use PyTorch + Metal (GPU) on Apple Silicon for maximum speed.
- If no `--piper-model` is provided, `en_US-libritts_r-medium.pt` is used by default.

> **Tip:** `BATCH_SIZE` only affects Piper TTS generation — higher values generate samples faster but use more memory.  
> **Tip:** If you rerun with the same wake word and sample count, it will **skip TTS generation** and use your cached clips, making retraining much faster.

