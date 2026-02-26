"""
Standalone configuration for the local voice pipeline.
All models run locally — no API keys needed for ASR or TTS.
"""
import os
import configparser

# Read API keys from orchestrator's config.ini (only for fallback/legacy)
_ORCHESTRATOR_INI = r"c:\Users\Administrator\genvoice-websocket-orchestrator\src\config\config.ini"
_config = configparser.ConfigParser()
_config.read(_ORCHESTRATOR_INI)


def _get(section: str, key: str, default: str = "") -> str:
    env_val = os.environ.get(key)
    if env_val:
        return env_val
    try:
        return _config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return default


# ----- ASR (Faster-Whisper — local GPU) -----
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3-turbo")  # or "medium", "small", "base"
ASR_DEVICE = os.environ.get("ASR_DEVICE", "cuda")
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "int8")  # int8 for speed, float16 for quality
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en")
ASR_SAMPLE_RATE = int(os.environ.get("ASR_SAMPLE_RATE", "16000"))

# ----- TTS (F5-TTS — local GPU, emotional) -----
TTS_MODEL_TYPE = os.environ.get("TTS_MODEL_TYPE", "F5TTS_v1_Base")
TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda")

# Reference audio for voice cloning
# REQUIREMENTS: Must be < 12 seconds, ref_text must EXACTLY match spoken words
# Default: built-in F5-TTS example voice (confirmed working)
# To use custom voice: set TTS_REF_AUDIO env var to your .wav path
from importlib.resources import files as _pkg_files
_DEFAULT_REF_AUDIO = str(_pkg_files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
_DEFAULT_REF_TEXT = "some call me nature, others call me mother nature."

TTS_REF_AUDIO = os.environ.get("TTS_REF_AUDIO", r"C:\\Users\Administrator\\local-voice-pipeline\\1-softever_Uwrt9r7Q.wav")
TTS_REF_TEXT = os.environ.get("TTS_REF_TEXT", "My love, when I look at you, I see not just a man but the only soul bold enough to stand before thunder and smile, the one who steadies my storms")
TTS_SAMPLE_RATE = 24000  # F5-TTS output rate
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.0"))  # 1.0 = natural, lower = slower
TTS_NFE_STEP = int(os.environ.get("TTS_NFE_STEP", "32"))  # 32 for clear speech

# ----- LLM (Local OpenAI-compatible API — Ollama) -----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")

# ----- Server -----
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8890"))
