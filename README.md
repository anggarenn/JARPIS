# JARPIS

<p align="center">
  <img src="https://raw.githubusercontent.com/anggarenn/JARPIS/main/assets/banner.png" alt="JARPIS Banner" width="100%" onerror="this.style.display='none'"/>
</p>

<p align="center">
  <strong>Ultra-Lightweight, Sub-800ms Bidirectional Voice Assistant for Local AI Agents (Zero-CUDA, Zero-Bloat)</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python" alt="Python 3.10+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License: MIT"/></a>
  <a href="#architecture"><img src="https://img.shields.io/badge/Hardware-Zero--CUDA-brightgreen.svg?style=flat-square" alt="Zero CUDA"/></a>
  <a href="#benchmarks"><img src="https://img.shields.io/badge/Latency-Sub--800ms-orange.svg?style=flat-square" alt="Sub-800ms Latency"/></a>
  <a href="#platform"><img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg?style=flat-square" alt="Platform"/></a>
</p>

---

## ⚡ Overview

**JARPIS** is a high-performance, bidirectional voice interaction system engineered specifically for local AI coding agents and developer workflows. Traditional voice assistants are either bloated with gigabytes of PyTorch CUDA dependencies or crippled by high-latency monolithic pipelines that force users to wait several seconds for speech generation.

JARPIS solves this through **streaming sentence boundary chunking**, an **in-memory zero-disk audio pipeline**, and **push-to-talk trigger precision**, achieving a **sub-800ms Time-to-First-Audio (TTFA)** on modest commodity hardware without requiring discrete NVIDIA GPUs.

---

## 🚀 Key Highlights

- **⚡ Sub-800ms Latency (TTFA):** Concurrently streams token deltas from your LLM gateway, chunks sentences on lexical boundaries, and pipes audio directly to non-blocking output queues while inference is still running.
- **🖥️ Zero-CUDA & Zero-Bloat:** Built for CPU inference (AVX2/FMA3) and lightweight ONNX/WebSocket runtimes. Requires **< 150 MB RAM** idle and eliminates heavyweight PyTorch/CUDA runtime bloat (> 4 GB saved).
- **🔒 Zero-Disk & Anti-Leak Privacy:** In-memory audio buffers (`io.BytesIO`) ensure zero microphone audio is written to disk. Audio buffers are zeroed out and garbage-collected immediately after transcription.
- **🎙️ Zero-Distraction Triggering:** Push-to-Talk (PTT default: `F8`) eliminates false triggers from background conversations, typing, or audio playback. Microphone capture handles remain closed when idle.
- **🌐 Hybrid Fallback Resilience:** Groq Cloud Whisper API primary STT (~120ms) with seamless auto-degradation to offline `faster-whisper` (CTranslate2 INT8 CPU). Primary Edge-TTS with offline Kokoro ONNX backup.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Capture["1. Audio Capture (Zero-Disk)"]
        User(["Developer / User"]) -->|Press & Hold F8| Mic["Sounddevice Mic Stream"]
        Mic -->|PCM 16kHz Mono| RAMBuf["In-Memory BytesIO Buffer"]
        User -->|Release F8| RAMBuf
    end

    subgraph STT["2. Speech-to-Text Pipeline"]
        RAMBuf --> RouterSTT{"Primary Online?"}
        RouterSTT -->|Yes| GroqSTT["Groq Whisper API (~120ms)"]
        RouterSTT -->|Fallback| LocalSTT["faster-whisper CPU INT8 (~420ms)"]
        GroqSTT --> PromptText["Transcribed Text Prompt"]
        LocalSTT --> PromptText
        RAMBuf -.->|Immediate Purge| RAMBuf
    end

    subgraph Brain["3. LLM Gateway & Chunker"]
        PromptText --> Gateway["OpenAI-Compatible Gateway / 9Router"]
        Gateway -->|Token Stream| Chunker["Sentence Boundary Chunker"]
    end

    subgraph TTS["4. Speech Synthesis & Playback"]
        Chunker -->|Sentence Chunks| EngineTTS{"Engine"}
        EngineTTS -->|Cloud Primary| EdgeTTS["Microsoft Edge Neural TTS"]
        EngineTTS -->|Offline Standby| KokoroTTS["Kokoro-82M ONNX"]
        EdgeTTS --> AudioQ["Non-Blocking Audio Player"]
        KokoroTTS --> AudioQ
        AudioQ --> Speaker(["Audio Output / Speaker (TTFA < 800ms)"])
    end
```

---

## 📊 Feature Matrix

| Feature / Metric | Legacy Python Assistants (SAPI5/pyttsx3) | Heavyweight AI Frameworks (AutoGPT / OpenJarvis) | **JARPIS** |
| :--- | :--- | :--- | :--- |
| **Time-to-First-Audio (TTFA)** | > 3500ms (blocking batch) | 1800ms – 4000ms | **< 800ms (streaming chunked)** |
| **CUDA / GPU Requirement** | None (CPU SAPI5) | Mandatory (NVIDIA CUDA, 4GB+ VRAM) | **Zero-CUDA (100% CPU / APU Native)** |
| **Memory Footprint** | ~100 MB | 2.5 GB – 6.0 GB | **< 150 MB (idle) / < 1.2 GB (peak)** |
| **Disk Dependency Bloat** | Low | Very High (> 4.5 GB PyTorch/TorchAudio) | **Ultra-Lean (< 350 MB total venv)** |
| **Audio Privacy** | Temporary `.wav` dumps | File system cached | **100% In-Memory (Zero-Disk Retention)** |
| **Trigger Control** | Always-listening (high CPU) | Wake-word (prone to false positives) | **Push-to-Talk (0% CPU idle) + Optional VAD** |
| **Voice Articulation** | Robotic SAPI5 | Variable | **Natural Neural Stream (`id-ID`, `en-US`)** |

---

## ⚡ Quick Start (3 Steps)

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/anggarenn/JARPIS.git
cd JARPIS

# Create virtual environment (Python 3.10+)
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

### Step 2: Install Lean Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> JARPIS does **not** install `torch` or `torchvision` by default, keeping your environment clean, lean, and fast.

### Step 3: Configure & Launch

```bash
# Copy environment configuration
cp .env.example .env

# Edit .env (set your LLM endpoint, model, or Groq API key)
```

Choose your preferred operation mode:

#### 🌐 Option A: Holographic Arc Reactor Web HUD (Recommended)
Launch the local web & WebSocket telemetry server and open the Iron Man 3 Second Screen HUD:
```bash
# Double-click or run:
start_ui.bat
# Or via CLI:
python run_server.py
```
Open **`http://127.0.0.1:20130`** in any desktop or mobile browser. Tap the Arc Reactor to speak or hold **`F8`** / **`ENTER`**!

#### 🛡️ Option B: Windows System Tray Background Daemon
Run JARPIS silently in the Windows Notification Area with a dynamic Stark Arc Reactor icon:
```bash
# Double-click or run:
start_tray.bat
# Or via CLI:
python run_tray.py
```
Right-click the tray icon to open the HUD, trigger listening, or exit.

#### 💻 Option C: Classic Interactive Terminal
Run in the terminal with Push-to-Talk console output:
```bash
# Double-click or run:
start.bat
# Or via CLI:
python run.py
```
Hold **`F8`** (or tap **`ENTER`** to toggle), speak your command, and release. Audio playback starts within 800ms!

---

## ⚙️ Configuration Guide

Configuration can be specified in `.env` or passed via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LLM_BASE_URL` | `http://127.0.0.1:20128/v1` | OpenAI-compatible LLM endpoint (9Router, Ollama, vLLM, LM Studio) |
| `LLM_MODEL` | `combo-hemat-coding` | Model identifier used for inference |
| `LLM_API_KEY` | *(empty)* | Optional API token for your LLM gateway |
| `JARPIS_PORT` | `20130` | Local port for Web HUD & WebSocket telemetry server |
| `GROQ_API_KEY` | *(empty)* | Groq API key for ultra-fast primary STT (~120ms) |
| `GROQ_MODEL` | `whisper-large-v3-turbo` | Groq Whisper model identifier |
| `TTS_VOICE` | `en-GB-RyanNeural` | British Paul Bettany JARVIS voice profile |
| `TTS_RATE` | `+3%` | Voice articulation speed adjustment |
| `TTS_PITCH` | `-5Hz` | Voice pitch calibration |
| `PTT_KEY` | `f8` | Global keyboard shortcut for Push-to-Talk (also supports `ENTER`) |
| `ENABLE_AUDIO_CUES` | `true` | Plays sci-fi dual-harmonic activation & send chimes |

---

## 🛡️ Privacy & Security Statement

JARPIS is designed from the ground up with defensive zero-trust audio principles:

1. **Zero Passive Listening:** In Push-to-Talk mode (default), the operating system microphone handle is kept closed until the physical key (`F8`) is pressed. When you are not speaking, JARPIS consumes 0% audio input and 0% CPU.
2. **Strict In-Memory Lifecycle:** Captured audio frames reside strictly in temporary RAM (`io.BytesIO`). No temporary `.wav` or `.mp3` files are ever created on your storage drive. Buffers are overwritten and garbage-collected immediately upon completion of transcription.
3. **No Unsanitized Shell Execution:** Voice commands are processed as structured prompt payloads to your agent brain rather than raw shell commands, preventing prompt injection attacks against your local system.

---

## 🤝 Contributing

Contributions, bug reports, and optimizations are welcome! Please follow these steps:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more details.

Copyright (c) 2026 **Angga Reno**.
