"""
Standalone configuration for the local voice pipeline.
All models run locally — no API keys needed for ASR or TTS.
"""
import os

# ----- ASR (Faster-Whisper — local GPU) -----
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3-turbo")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "float16")  # float16 for accuracy
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en")
ASR_SAMPLE_RATE = int(os.environ.get("ASR_SAMPLE_RATE", "16000"))

# ----- TTS (Kokoro — ultra-fast local GPU) -----
TTS_VOICE = os.environ.get("TTS_VOICE", "jf_alpha")  # Japanese female anime waifu voice
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.0"))  # 1.0 = normal
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda")
TTS_SAMPLE_RATE = 24000

# ----- LLM (Local — Ollama) -----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:1b")

# ----- Server -----
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8890"))
