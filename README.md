# Local Voice Pipeline

Ultra-Fast Local Voice Pipeline that enables end-to-end true voice communication using offline AI models. 
It supports both a Browser WebSocket-based UI and a Terminal-Only Mode.

## Features
- **ASR (Speech-to-Text):** Faster-Whisper (local GPU) for ultra-fast, accurate transcription.
- **VAD (Voice Activity Detection):** Silero VAD for highly responsive speech detection.
- **LLM (Language Model):** Local OpenAI-compatible API (Ollama/llama.cpp) with GGUF models. It streams responses sentence-by-sentence.
- **TTS (Text-to-Speech):** Supports multiple engines:
  - Kokoro TTS
  - F5-TTS
  - Fish-Speech
- **Barge-In Support:** Interrupt the AI while it's speaking by simply talking over it.
- **Interfaces:**
  - `server.py`: FastAPI + WebSocket server for browser-based UI.
  - `terminal_app.py`: Rich-based terminal UI for mic/speaker integration directly in the console.

## Setup
Ensure you have the required dependencies:
```bash
pip install -r requirements.txt
```
Set up the `.env` file with any API keys or configuration options following the defined config settings in `config.py`.

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
3. Audio is sent to Faster-Whisper for transcription.
4. Transcript is sent to the local LLM.
5. The LLM streams sentences back.
6. Each sentence is synthesized into audio chunks using Kokoro/F5-TTS/Fish-Speech.
7. Audio chunks are played gaplessly (either via browser Web Audio API or sounddevice in terminal).
