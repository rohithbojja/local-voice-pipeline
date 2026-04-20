# Local Voice Pipeline

Ultra-Fast Local Voice Pipeline that enables end-to-end true voice communication using offline AI models. 
It supports both a Browser WebSocket-based UI and a Terminal-Only Mode.

## Features
- **ASR (Speech-to-Text):** [Parakeet TDT MLX](https://huggingface.co/animaslabs/parakeet-tdt-0.6b-v3-mlx-4bit) (Apple Silicon / MLX, 4-bit quantized).
- **VAD (Voice Activity Detection):** Silero VAD for highly responsive speech detection.
- **LLM (Language Model):** Local OpenAI-compatible API (Ollama/llama.cpp) with GGUF models. It streams responses sentence-by-sentence.
- **TTS (Text-to-Speech):** [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) (local, multilingual; always clones from your reference clip).
- **Barge-In Support:** Interrupt the AI while it's speaking by simply talking over it.
- **Interfaces:**
  - `server.py`: FastAPI + WebSocket server for browser-based UI.
  - `terminal_app.py`: Rich-based terminal UI for mic/speaker integration directly in the console.

## Setup
Create a virtual environment and install dependencies (recommended: [uv](https://github.com/astral-sh/uv)):
```bash
uv venv && source .venv/bin/activate
uv sync
```
Copy **`.env.example`** to **`.env`** and adjust (Ollama model name, etc.). **ASR** requires **Apple Silicon** and **`parakeet-mlx`**; override **`ASR_MODEL`** if you use another Parakeet MLX checkpoint on the Hub. **ffmpeg** must be on `PATH` (used by Parakeet to decode audio). TTS clones from **`reference.wav`** + **`reference.txt`** next to `config.py` unless overridden.

## Usage

### 1. Browser Interface (WebSocket)
Run the server:
```bash
python server.py
```
Then open `http://localhost:8890` (or the port defined in config) in your browser to interact with the pipeline.

### 2. Terminal UI
Run the terminal app directly:
```bash
python terminal_app.py
```
You can speak into your microphone and the AI will respond through your speakers directly in the terminal.

## Architecture
1. Microphone captures audio.
2. Silero VAD detects voice activity.
3. Audio is transcribed with Parakeet TDT (MLX).
4. Transcript is sent to the local LLM.
5. The LLM streams sentences back.
6. Each sentence is synthesized with OmniVoice and streamed as PCM audio.
7. Audio chunks are played gaplessly (either via browser Web Audio API or sounddevice in terminal).
