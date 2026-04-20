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


# ----- ASR (Parakeet TDT — MLX on Apple Silicon only; HF model id) -----
# Default: https://huggingface.co/animaslabs/parakeet-tdt-0.6b-v3-mlx-4bit
ASR_MODEL = os.environ.get(
    "ASR_MODEL", "animaslabs/parakeet-tdt-0.6b-v3-mlx-4bit"
)
ASR_SAMPLE_RATE = int(os.environ.get("ASR_SAMPLE_RATE", "16000"))

# ----- TTS (OmniVoice — https://huggingface.co/k2-fsa/OmniVoice) -----
OMNIVOICE_MODEL = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
# Always voice-clone: default ``reference.wav`` + ``reference.txt`` beside this file, or env.
OMNIVOICE_REF_AUDIO, OMNIVOICE_REF_TEXT = _resolve_omnivoice_reference()
# cuda | mps | cpu (mapped to cuda:0 / mps:0 / cpu inside the runner)
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda")
# Language hint for better quality (e.g. "en", "English"); empty = model default / agnostic
OMNIVOICE_LANGUAGE = os.environ.get("OMNIVOICE_LANGUAGE", "en").strip() or None
_omni_speed = os.environ.get("OMNIVOICE_SPEED", "").strip()
OMNIVOICE_SPEED: float | None = float(_omni_speed) if _omni_speed else None
TTS_SAMPLE_RATE = 24000  # OmniVoice is typically 24 kHz; runner updates from the model after load

# ----- LLM (Local — Ollama) -----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4-obliterated:latest")
# OpenAI-compatible: "none" turns off Gemma 4 style thinking on supported Ollama builds.
# Set empty to omit the field (older Ollama).
LLM_REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "none").strip() or None

# ----- Server -----
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8890"))
