"""
Standalone configuration for the local voice pipeline.
Reads API keys directly from the orchestrator's config.ini — NO orchestrator code imports.
"""
import os
import configparser

# Read API keys from orchestrator's config.ini (just the INI file, not the code)
_ORCHESTRATOR_INI = r"c:\Users\Administrator\genvoice-websocket-orchestrator\src\config\config.ini"
_config = configparser.ConfigParser()
_config.read(_ORCHESTRATOR_INI)


def _get(section: str, key: str, default: str = "") -> str:
    """Get a config value from ini, with env var override."""
    env_val = os.environ.get(key)
    if env_val:
        return env_val
    try:
        return _config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return default


# ----- ASR (Microsoft Azure Speech SDK) -----
MICROSOFT_KEY = _get("microsoft_config", "MICROSOFT_TTS_KEY")
MICROSOFT_REGION = _get("microsoft_config", "MICROSOFT_TTS_REGION", "centralindia")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en-IN")
ASR_SAMPLE_RATE = int(os.environ.get("ASR_SAMPLE_RATE", "16000"))

# ----- TTS (ElevenLabs) -----
ELEVENLABS_API_KEY = _get("elevenlabs_config", "ELEVENLABS_API_KEY_IN")
ELEVENLABS_BASE_URL = _get("elevenlabs_config", "ELEVENLABS_BASE_URL", "https://api.in.residency.elevenlabs.io")
TTS_VOICE = os.environ.get("TTS_VOICE", "EXAVITQu4vr4xnSDxMaL")  # "Bella" — soft, young female
TTS_MODEL = os.environ.get("TTS_MODEL", "eleven_turbo_v2_5")
TTS_SAMPLE_RATE = 24000  # ElevenLabs PCM output rate

# ----- LLM (Local OpenAI-compatible API) -----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")

# ----- Server -----
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8890"))
