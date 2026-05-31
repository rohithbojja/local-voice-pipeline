"""
Standalone configuration for the local voice pipeline.
ASR and TTS run locally; LLM uses a local OpenAI-compatible API (e.g. Ollama).
"""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# Load `.env` from project root (non-fatal if python-dotenv missing).
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass


# ----- ASR (Faster-Whisper — local) -----
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3-turbo")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "float16")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en")
ASR_SAMPLE_RATE = int(os.environ.get("ASR_SAMPLE_RATE", "16000"))
ASR_CPU_THREADS = int(os.environ.get("ASR_CPU_THREADS", "0"))

# ----- TTS (Supertonic — https://github.com/supertone-inc/supertonic) -----
# pip install 'supertonic[serve]'
SUPERTONIC_VOICE = os.environ.get("SUPERTONIC_VOICE", "F1")
# Optional: path to a Voice Builder JSON for custom voice cloning.
# Download from https://supertonic.supertone.ai/voice-builder
# When set, overrides SUPERTONIC_VOICE.
SUPERTONIC_VOICE_JSON: str | None = os.environ.get("SUPERTONIC_VOICE_JSON", "").strip() or None
SUPERTONIC_LANG = os.environ.get("SUPERTONIC_LANG", "en")
_st_speed = os.environ.get("SUPERTONIC_SPEED", "1.05").strip()
SUPERTONIC_SPEED: float = float(_st_speed) if _st_speed else 1.05
_st_steps = os.environ.get("SUPERTONIC_STEPS", "5").strip()   # 2=fastest 5=balanced 10=high
SUPERTONIC_STEPS: int = int(_st_steps) if _st_steps else 5
_st_silence = os.environ.get("SUPERTONIC_SILENCE_DURATION", "0.15").strip()
SUPERTONIC_SILENCE_DURATION: float = float(_st_silence) if _st_silence else 0.15
_st_intra = os.environ.get("SUPERTONIC_INTRA_OP_THREADS", "").strip()
SUPERTONIC_INTRA_OP_THREADS: int | None = int(_st_intra) if _st_intra else None
_st_inter = os.environ.get("SUPERTONIC_INTER_OP_THREADS", "").strip()
SUPERTONIC_INTER_OP_THREADS: int | None = int(_st_inter) if _st_inter else None
TTS_SAMPLE_RATE = 44100  # Supertonic outputs 44.1 kHz

# ----- LLM (Local — Ollama) -----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:1b")

# ----- Server -----
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8890"))
