"""
TTS Runner — Kokoro TTS (ultra-fast local GPU inference).
82M params, sub-300ms latency, 24kHz output.
"""
import logging
import numpy as np
import torch
import asyncio
import re

logger = logging.getLogger(__name__)


class TTSRunner:
    """
    Kokoro TTS — blazingly fast local TTS.
    82M params, ~200ms inference on GPU, 24kHz output.
    """

    def __init__(self, voice: str = "af_heart", device: str = None, speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = None
        self.sample_rate = 24000

    def _ensure_model(self):
        if self._pipeline is None:
            logger.info(f"Loading Kokoro TTS (voice={self.voice}) on {self.device}...")
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code='a', device=self.device)
            logger.info("Kokoro TTS loaded")

    async def start(self):
        """Preload the model."""
        await asyncio.to_thread(self._ensure_model)

    def generate(self, text: str) -> bytes:
        """Generate PCM16 audio bytes from text. Ultra-fast."""
        self._ensure_model()
        try:
            text = self._clean_text(text)
            logger.info(f"TTS generating: {text[:80]}...")

            # Kokoro generates audio segments
            audio_segments = []
            for result in self._pipeline(text, voice=self.voice, speed=self.speed):
                if result.audio is not None:
                    audio_segments.append(result.audio.numpy())

            if not audio_segments:
                logger.warning("TTS produced no audio")
                return b""

            # Concatenate all segments
            wav = np.concatenate(audio_segments)

            # Normalize
            max_val = np.max(np.abs(wav))
            if max_val > 0:
                wav = wav / max_val * 0.95

            pcm16 = (wav * 32767).astype(np.int16)
            audio_bytes = pcm16.tobytes()
            duration = len(pcm16) / self.sample_rate
            logger.info(f"TTS generated {len(audio_bytes)} bytes ({duration:.1f}s)")
            return audio_bytes

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return b""

    def generate_streaming(self, text: str):
        """Generate audio chunks as a generator — for sentence-level streaming."""
        self._ensure_model()
        text = self._clean_text(text)

        for result in self._pipeline(text, voice=self.voice, speed=self.speed):
            if result.audio is not None:
                wav = result.audio.numpy()
                max_val = np.max(np.abs(wav))
                if max_val > 0:
                    wav = wav / max_val * 0.95
                pcm16 = (wav * 32767).astype(np.int16)
                yield pcm16.tobytes()

    def _clean_text(self, text: str) -> str:
        """Strip emojis, markdown, and other TTS-unfriendly chars."""
        # Remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0\U000024C2-\U0001F251"
            "\U00010000-\U0010ffff\u200d\ufe0f"
            "\u2600-\u26FF\u2700-\u27BF"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub("", text)
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'_+', ' ', text)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def cleanup(self):
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
