"""
ASR Runner — Faster-Whisper (local GPU inference, no API key needed).
Processes PCM16 audio chunks and yields transcripts.
Only transcribes when VAD signals speech_end — no background loop.

IMPORTANT: Always buffers audio with a rolling pre-buffer so we don't
miss the beginning of speech before VAD fires speech_start.
"""
import asyncio
import logging
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ASRRunner:
    """
    Faster-Whisper ASR — runs locally on GPU.
    Always keeps a rolling pre-buffer of recent audio so we capture
    speech from BEFORE the VAD detects it.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        sample_rate: int = 16000,
        cpu_threads: int = 0,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.sample_rate = sample_rate
        self.cpu_threads = cpu_threads
        self._model = None
        self._transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._running = False
        self._loop = None
        self._speech_active = False

        # Rolling pre-buffer: keeps last 1 second of audio so we capture
        # speech that started BEFORE VAD detected it
        self._pre_buffer_seconds = 1.0
        self._pre_buffer_size = int(sample_rate * self._pre_buffer_seconds)
        self._pre_buffer = np.array([], dtype=np.float32)

        # Speech buffer: audio accumulated during active speech
        self._speech_buffer = np.array([], dtype=np.float32)

        self._min_audio_length = 0.3  # Min seconds to attempt transcription

    def _ensure_model(self):
        if self._model is None:
            logger.info(f"Loading Whisper model: {self.model_size} ({self.compute_type} on {self.device})...")
            kw = {"device": self.device, "compute_type": self.compute_type}
            if self.device == "cpu" and self.cpu_threads > 0:
                kw["cpu_threads"] = self.cpu_threads
            self._model = WhisperModel(self.model_size, **kw)
            logger.info("Whisper model loaded")

    def set_speech_active(self, active: bool):
        """Called by VAD to track speech state."""
        if active and not self._speech_active:
            # Speech just started — copy pre-buffer into speech buffer
            # so we have the audio from BEFORE VAD detected speech
            self._speech_buffer = self._pre_buffer.copy()
            logger.debug(f"Speech started: pre-buffered {len(self._speech_buffer)/self.sample_rate:.2f}s")
        self._speech_active = active

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM16 audio bytes. Always buffers for pre-buffer;
        accumulates into speech buffer during active speech."""
        if not self._running:
            return
        # Convert PCM16 bytes → float32 [-1, 1]
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
        float32 = pcm16.astype(np.float32) / 32768.0

        # Always maintain rolling pre-buffer (last N seconds)
        self._pre_buffer = np.concatenate([self._pre_buffer, float32])
        if len(self._pre_buffer) > self._pre_buffer_size:
            self._pre_buffer = self._pre_buffer[-self._pre_buffer_size:]

        # Accumulate into speech buffer during active speech
        if self._speech_active:
            self._speech_buffer = np.concatenate([self._speech_buffer, float32])

    def transcribe_buffer(self):
        """Transcribe accumulated speech audio. Called ONLY on VAD speech_end."""
        if self._model is None or len(self._speech_buffer) < int(self.sample_rate * self._min_audio_length):
            self._speech_buffer = np.array([], dtype=np.float32)
            return ""

        audio = self._speech_buffer.copy()
        self._speech_buffer = np.array([], dtype=np.float32)

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
        if len(t) < 3:
            return True
        hallucination_markers = [
            "thank you for watching", "thanks for watching",
            "subscribe", "like and subscribe", "please subscribe",
        ]
        for marker in hallucination_markers:
            if marker in t:
                return True
        words = t.split()
        if len(words) >= 6:
            for i in range(len(words) - 5):
                phrase = " ".join(words[i:i+3])
                if t.count(phrase) >= 3:
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
        self._speech_buffer = np.array([], dtype=np.float32)
        self._pre_buffer = np.array([], dtype=np.float32)
        logger.info("ASR stopped")
