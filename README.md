<div align="center">
  <img src="https://raw.githubusercontent.com/TaterTotterson/microWakeWord-Trainer-Docker/refs/heads/main/mmw.png" alt="MicroWakeWord Trainer Logo" width="100" />
  <h1>microWakeWord Trainer Apple Silicon</h1>
</div>

## **Quick Start:**

### Make the training script executable (first time only)
```bash
chmod +x train_microwakeword_macos.sh
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
### Train with a custom Piper voice
```bash
./train_microwakeword_macos.sh "hey_phooey" 50000 100 --piper-model voices/en_US-amy.pt
```
### Multi-Voice Training (Recommended)

You can train with multiple Piper voices to make the model more robust across speakers.
Pass `--piper-model` **multiple times** (accepts `.pt` or `.onnx`):

./train_microwakeword_macos.sh "hey_tater" 50000 100 \
  --piper-model piper-sample-generator/models/en_US-libritts_r-medium.pt \
  --piper-model voices/en_US-amy.pt \
  --piper-model voices/en_GB-male.onnx

Notes:
- Both `.pt` **and** `.onnx` Piper models are supported and can be mixed freely.
- If no `--piper-model` is provided, a default English voice is auto-downloaded.
- Using 2–3 diverse voices is often enough — too many can slow sample generation.
- See a list of official Piper voices here: https://huggingface.co/rhasspy/piper-voices/tree/main  
- `BATCH_SIZE` (3rd arg) only affects Piper TTS generation throughput/memory, not TF training.

> **Tip:** `BATCH_SIZE` only affects Piper TTS generation — higher values generate samples faster but use more memory.  
> **Tip:** If you rerun with the same wake word and sample count, it will **skip TTS generation** and use your cached clips, making retraining much faster.

