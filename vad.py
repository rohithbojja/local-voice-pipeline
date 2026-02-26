"""
VAD (Voice Activity Detection) — Standalone Silero VAD wrapper.
Detects speech start/stop from raw PCM16 audio chunks.
"""
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Silero VAD model (loaded once, shared across sessions)
_vad_model = None
_vad_utils = None


def _load_model():
    """Lazy load Silero VAD model."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        logger.info("Loading Silero VAD model...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _vad_model = model
        _vad_utils = utils
        logger.info("Silero VAD model loaded")
    return _vad_model


class VADProcessor:
    """
    Processes audio chunks and emits speech start/stop events.
    Uses Silero VAD for accurate voice activity detection.
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5,
                 min_speech_ms: int = 250, min_silence_ms: int = 700):
        """
        Args:
            sample_rate: Audio sample rate (must be 16000 for Silero)
            threshold: VAD confidence threshold (0.0-1.0)  
            min_speech_ms: Minimum speech duration to trigger "started speaking"
            min_silence_ms: Minimum silence duration to trigger "stopped speaking"
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_ms / 1000)

        self.model = _load_model()
        self._is_speaking = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._audio_buffer = np.array([], dtype=np.int16)

        # Silero VAD processes 512-sample windows at 16kHz (32ms)
        self.window_size = 512

    def reset(self):
        """Reset VAD state for a new session."""
        self._is_speaking = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._audio_buffer = np.array([], dtype=np.int16)
        self.model.reset_states()

    def process(self, audio_bytes: bytes) -> list[dict]:
        """
        Process audio chunk and return list of events.
        
        Events:
            {"type": "speech_start"} — user started speaking
            {"type": "speech_end", "audio": bytes} — user stopped, includes buffered speech audio
        
        Returns:
            List of event dicts (may be empty if no state change).
        """
        events = []

        # Convert bytes → int16 numpy array
        chunk = np.frombuffer(audio_bytes, dtype=np.int16)
        self._audio_buffer = np.concatenate([self._audio_buffer, chunk])

        # Process in 512-sample windows
        while len(self._audio_buffer) >= self.window_size:
            window = self._audio_buffer[:self.window_size]
            self._audio_buffer = self._audio_buffer[self.window_size:]

            # Convert to float32 tensor for Silero
            tensor = torch.from_numpy(window.astype(np.float32) / 32768.0)

            # Get VAD confidence
            with torch.no_grad():
                confidence = self.model(tensor, self.sample_rate).item()

            is_speech = confidence > self.threshold

            if is_speech:
                self._speech_samples += self.window_size
                self._silence_samples = 0

                if not self._is_speaking and self._speech_samples >= self.min_speech_samples:
                    self._is_speaking = True
                    events.append({"type": "speech_start"})
                    logger.info(f"VAD: Speech started (confidence={confidence:.2f})")
            else:
                self._silence_samples += self.window_size

                if self._is_speaking and self._silence_samples >= self.min_silence_samples:
                    self._is_speaking = False
                    self._speech_samples = 0
                    events.append({"type": "speech_end"})
                    logger.info(f"VAD: Speech ended (silence={self._silence_samples/self.sample_rate:.2f}s)")

        return events

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
