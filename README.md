<div align="center">
  <h1>🎙️ microWakeWord AppleSilicon Trainer & Recorder</h1>
  <img width="990" height="582" alt="Screenshot 2026-01-15 at 10 02 28 PM" src="https://github.com/user-attachments/assets/335cb187-75e6-46f7-abb5-dfe2f3456b14" />
</div>

---

This project lets you **create custom wake words** for Home Assistant Voice using a combination of:

- **Local voice recordings** (your real voice, optional but recommended)
- **Automatically generated TTS samples**
- A **fully automated training pipeline**

You can either:
1. Use the **local Web UI** to record real voice samples and auto-train  
2. Or run the **training script directly** (TTS-only or with pre-existing samples)

---

## 🧪 Important First Step — Test Your Wake Word with TTS

Before recording or training anything:

> **Test your wake word with text-to-speech first.**

Some words or names are pronounced differently by TTS engines.  
You may need to **spell the word creatively** (for example: `tay-ter` instead of `tater`)
to get consistent pronunciation.

The Web UI includes a **🔊 Test TTS** button for this reason.

---
> **Note:** The script will automatically install **ffmpeg** and **wget** via Homebrew
if they are missing. Homebrew itself must already be installed:
https://brew.sh/

## **Clone Repo:**
Clone the repo and enter the folder:
```bash
git clone https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon.git
cd microWakeWord-Trainer-AppleSilicon
```
---

## 🚀 Option 1: Run the Web UI (Recommended)

The Web UI guides users through:
- Entering a wake word
- Testing TTS pronunciation
- Recording real voice samples (auto-start / auto-stop)
- Supporting **multiple speakers** (family members)
- Automatically starting training when recordings are complete

### ▶️ Start the Recorder Web UI

From the project root:

```bash
./run_recorder_macos.sh
```

What this does:
- Creates and manages `.recorder-venv`
- Installs all required dependencies (once)
- Starts a local FastAPI server with the recording UI

Then open your browser to:

```
http://127.0.0.1:8789
```

---

### 🎙️ Recording Flow

1. Enter your wake word
2. Test pronunciation with **Test TTS**
3. Choose:
   - Number of speakers (e.g. family members)
   - Takes per speaker (default: 10)
4. Click **Begin recording**
5. Speak naturally — recording:
   - Starts when you talk
   - Stops automatically after silence
6. Repeat for each speaker

Files are saved automatically to:

```
personal_samples/
  speaker01_take01.wav
  speaker01_take02.wav
  speaker02_take01.wav
  ...
```

> ⚠️ The training pipeline automatically detects **any `.wav` files** in
> `personal_samples/` and gives them extra weight over TTS samples.

---

### 🧠 Automatic Training

Once **all recordings are finished**:
- The microphone is stopped
- Training starts automatically
- Live training logs stream into the Web UI

Reloading the page **does NOT interrupt training** — it continues in the background.

---

## 🧪 Option 2: Run Training Script Only (No Web UI)

If you don’t want to record real voice samples, or you already have them, you can run training directly.

### ▶️ Basic Training (TTS-only)

```bash
./train_microwakeword_macos.sh "hey_tater"
```

This will:
- Create/use `.venv`
- Generate TTS samples
- Train a wake word model
- Output the final model file

---

### 🎙️ Training with Personal Voice Samples

If **any `.wav` files exist** in:

```
personal_samples/
```

They are automatically included and weighted higher than TTS samples.

No flags required — the script detects them automatically.

---

## ⚠️ Notes

- Please use **one wake word per training run**
- Avoid punctuation or emojis in wake words
- Training runs **sequentially**
- Multiple speakers improve real-world detection accuracy
- Page reloads do **not** interrupt training

---

## 🧩 When to Use Each Mode

| Use case | Recommended path |
|--------|------------------|
| Best accuracy | Web UI + real voice recordings |
| Quick testing | Training script only |
| Family / shared device | Web UI with multiple speakers |
| Headless / CI | Training script only |

---
