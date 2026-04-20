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


def _resolve_omnivoice_reference() -> tuple[str, str]:
    """Voice clone is always on: WAV path + transcript (env or ``reference.txt``)."""
    audio = (os.environ.get("OMNIVOICE_REF_AUDIO") or str(_ROOT / "reference.wav")).strip()
    text = os.environ.get("OMNIVOICE_REF_TEXT", "").strip()
    if not text:
        txt_path = _ROOT / "reference.txt"
        if txt_path.is_file():
            text = txt_path.read_text(encoding="utf-8").strip()
    return audio, text


# ----- ASR (Faster-Whisper — local) -----
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3-turbo")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "float16")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en")
ASR_SAMPLE_RATE = int(os.environ.get("ASR_SAMPLE_RATE", "16000"))
ASR_CPU_THREADS = int(os.environ.get("ASR_CPU_THREADS", "0"))

# ----- TTS (OmniVoice — https://huggingface.co/k2-fsa/OmniVoice) -----
OMNIVOICE_MODEL = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
OMNIVOICE_REF_AUDIO, OMNIVOICE_REF_TEXT = _resolve_omnivoice_reference()
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda")
OMNIVOICE_LANGUAGE = os.environ.get("OMNIVOICE_LANGUAGE", "en").strip() or None
_omni_speed = os.environ.get("OMNIVOICE_SPEED", "").strip()
OMNIVOICE_SPEED: float | None = float(_omni_speed) if _omni_speed else None
TTS_SAMPLE_RATE = 24000

# ----- LLM (Local — Ollama) -----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2:1b")

# ----- Server -----
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8890"))
