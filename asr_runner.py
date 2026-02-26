"""
ASR Runner — Faster-Whisper (local GPU inference, no API key needed).
Processes PCM16 audio chunks and yields transcripts.
Only transcribes when VAD signals speech_end — no background loop.
"""
import asyncio
import logging
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ASRRunner:
    """
    Faster-Whisper ASR — runs locally on GPU.
    Accumulates audio chunks and transcribes ONLY when VAD detects speech end.
    No background interim loop — avoids Whisper hallucinations on silence.
    """

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cuda",
                 compute_type: str = "int8", language: str = "en", sample_rate: int = 16000):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.sample_rate = sample_rate
        self._model = None
        self._transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._audio_buffer = np.array([], dtype=np.float32)
        self._running = False
        self._loop = None
        self._speech_active = False  # Track VAD state
        self._min_audio_length = 0.5  # Min seconds of audio to transcribe

    def _ensure_model(self):
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_size} ({self.compute_type} on {self.device})...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Whisper model loaded")

    def set_speech_active(self, active: bool):
        """Called by VAD to track speech state. Only accumulate audio during speech."""
        self._speech_active = active

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM16 audio bytes into the buffer — only during active speech."""
        if not self._running or not self._speech_active:
            return
        # Convert PCM16 bytes → float32 [-1, 1]
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
        float32 = pcm16.astype(np.float32) / 32768.0
        self._audio_buffer = np.concatenate([self._audio_buffer, float32])

    def transcribe_buffer(self):
        """Transcribe accumulated audio. Called ONLY on VAD speech_end."""
        if self._model is None or len(self._audio_buffer) < int(self.sample_rate * self._min_audio_length):
            self._audio_buffer = np.array([], dtype=np.float32)
            return ""

        audio = self._audio_buffer.copy()
        self._audio_buffer = np.array([], dtype=np.float32)

        try:
            segments, info = self._model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()

            # Filter out known Whisper hallucination patterns
            if text and not self._is_hallucination(text):
                logger.info(f"ASR Final: {text}")
                try:
                    self._loop.call_soon_threadsafe(
                        self._transcript_queue.put_nowait,
                        {"type": "final", "text": text}
                    )
                except Exception as e:
                    logger.error(f"Error queuing transcript: {e}")
                return text
            else:
                if text:
                    logger.warning(f"ASR filtered hallucination: {text}")
                return ""
        except Exception as e:
            logger.error(f"Whisper transcribe error: {e}", exc_info=True)
            return ""

    def _is_hallucination(self, text: str) -> bool:
        """Detect common Whisper hallucination patterns."""
        t = text.lower().strip()
        # Too short or just filler
        if len(t) < 3:
            return True
        # Repeated "Thank you" at end is a classic hallucination
        hallucination_markers = [
            "thank you for watching",
            "thanks for watching",
            "subscribe",
            "like and subscribe",
            "please subscribe",
        ]
        for marker in hallucination_markers:
            if marker in t:
                return True
        # Excessive repetition (same phrase 3+ times)
        words = t.split()
        if len(words) >= 6:
            # Check for 3-word phrase repetition
            for i in range(len(words) - 5):
                phrase = " ".join(words[i:i+3])
                count = t.count(phrase)
                if count >= 3:
                    return True
        return False

    async def start(self):
        """Initialize model and return transcript queue."""
        self._loop = asyncio.get_event_loop()
        self._running = True
        await asyncio.to_thread(self._ensure_model)
        logger.info(f"ASR started ({self.model_size}, {self.compute_type}, lang={self.language})")
        return self._transcript_queue

    def stop(self):
        self._running = False
        self._audio_buffer = np.array([], dtype=np.float32)
        logger.info("ASR stopped")
