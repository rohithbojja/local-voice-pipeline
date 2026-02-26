"""
TTS Runner — Standalone ElevenLabs TTS using the elevenlabs SDK directly.
No orchestrator imports. Returns raw PCM16 audio bytes.
"""
import logging
from elevenlabs import ElevenLabs, VoiceSettings

logger = logging.getLogger(__name__)


class TTSRunner:
    """ElevenLabs TTS — standalone, generates PCM16 audio bytes from text."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.in.residency.elevenlabs.io",
        voice_id: str = "pNInz6obpgDQGcFmaJgB",
        model_id: str = "eleven_turbo_v2_5",
        output_format: str = "pcm_24000",  # 24kHz 16-bit PCM
    ):
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.client = ElevenLabs(api_key=api_key, base_url=base_url)
        logger.info(f"TTS initialized (voice={voice_id}, model={model_id}, format={output_format})")

    def generate(self, text: str) -> bytes:
        """Generate PCM audio bytes from text."""
        try:
            logger.info(f"TTS generating: {text[:80]}...")
            voice_settings = VoiceSettings(
                stability=0.7,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
                speed=1.0,
            )
            audio_chunks = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                output_format=self.output_format,
                text=text,
                model_id=self.model_id,
                voice_settings=voice_settings,
            )
            # Collect all chunks
            audio = b""
            for chunk in audio_chunks:
                audio += chunk
            logger.info(f"TTS generated {len(audio)} bytes")
            return audio
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""

    def cleanup(self):
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass
