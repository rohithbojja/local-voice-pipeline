"""
ASR Runner — Parakeet TDT (MLX, Apple Silicon) from Hugging Face.
Processes PCM16 audio chunks and yields transcripts on VAD speech_end.

Uses animaslabs/parakeet-tdt-0.6b-v3-mlx-4bit (or any compatible Parakeet MLX checkpoint).
See: https://huggingface.co/animaslabs/parakeet-tdt-0.6b-v3-mlx-4bit
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path

import librosa
import mlx.nn as nn
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

from parakeet_mlx.utils import from_config

logger = logging.getLogger(__name__)


def load_parakeet_mlx(model_id: str, cache_dir: str | Path | None = None):
    """
    Load a Parakeet MLX model from the Hub. Applies MLX quantization when
    ``quantization`` is present in ``config.json`` (4-bit / 8-bit checkpoints).
    """
    config_path = hf_hub_download(model_id, "config.json", cache_dir=cache_dir)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    model = from_config(config)
    q = config.get("quantization")
    if q:
        nn.quantize(model, bits=q["bits"], group_size=q["group_size"])
    weights = hf_hub_download(model_id, "model.safetensors", cache_dir=cache_dir)
    model.load_weights(weights)
    model.eval()
    return model


class ASRRunner:
    """
    Parakeet-TDT MLX ASR. Keeps a rolling pre-buffer so we do not miss
    speech that starts before VAD fires speech_start.
    """

    def __init__(self, model_id: str, sample_rate: int = 16000):
        self.model_id = model_id
        self.sample_rate = sample_rate  # Input PCM rate from the browser / VAD path
        self._model = None
        self._transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._running = False
        self._loop = None
        self._speech_active = False

        self._pre_buffer_seconds = 1.0
        self._pre_buffer_size = int(sample_rate * self._pre_buffer_seconds)
        self._pre_buffer = np.array([], dtype=np.float32)
        self._speech_buffer = np.array([], dtype=np.float32)

        self._min_audio_length = 0.3

    def _target_sample_rate(self) -> int:
        if self._model is None:
            return self.sample_rate
        return int(self._model.preprocessor_config.sample_rate)

    def _ensure_model(self):
        if self._model is None:
            logger.info("Loading Parakeet MLX: %s ...", self.model_id)
            self._model = load_parakeet_mlx(self.model_id)
            tsr = self._target_sample_rate()
            if tsr != self.sample_rate:
                logger.warning(
                    "Model expects %s Hz audio; client/VAD uses %s Hz — resampling on transcribe.",
                    tsr,
                    self.sample_rate,
                )
            logger.info("Parakeet MLX loaded (native %s Hz)", tsr)

    def set_speech_active(self, active: bool):
        if active and not self._speech_active:
            self._speech_buffer = self._pre_buffer.copy()
            logger.debug(
                "Speech started: pre-buffered %.2fs",
                len(self._speech_buffer) / self.sample_rate,
            )
        self._speech_active = active

    def feed_audio(self, audio_bytes: bytes):
        if not self._running:
            return
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
        float32 = pcm16.astype(np.float32) / 32768.0

        self._pre_buffer = np.concatenate([self._pre_buffer, float32])
        if len(self._pre_buffer) > self._pre_buffer_size:
            self._pre_buffer = self._pre_buffer[-self._pre_buffer_size :]

        if self._speech_active:
            self._speech_buffer = np.concatenate([self._speech_buffer, float32])

    def transcribe_buffer(self):
        if self._model is None or len(self._speech_buffer) < int(
            self.sample_rate * self._min_audio_length
        ):
            self._speech_buffer = np.array([], dtype=np.float32)
            return ""

        audio = self._speech_buffer.copy()
        self._speech_buffer = np.array([], dtype=np.float32)

        try:
            target_sr = self._target_sample_rate()
            if self.sample_rate != target_sr:
                audio = librosa.resample(
                    audio.astype(np.float32),
                    orig_sr=self.sample_rate,
                    target_sr=target_sr,
                )

            fd, path = tempfile.mkstemp(suffix=".wav")
            path = Path(path)
            try:
                os.close(fd)
                sf.write(str(path), audio, target_sr, subtype="PCM_16")
                result = self._model.transcribe(str(path))
                text = (result.text or "").strip()
            finally:
                path.unlink(missing_ok=True)

            if text and not self._is_hallucination(text):
                logger.info("ASR Final: %s", text)
                try:
                    self._loop.call_soon_threadsafe(
                        self._transcript_queue.put_nowait,
                        {"type": "final", "text": text},
                    )
                except Exception as e:
                    logger.error("Error queuing transcript: %s", e)
                return text
            if text:
                logger.warning("ASR filtered hallucination-like output: %s", text[:80])
            return ""
        except Exception as e:
            logger.error("Parakeet transcribe error: %s", e, exc_info=True)
            return ""

    def _is_hallucination(self, text: str) -> bool:
        t = text.lower().strip()
        if len(t) < 2:
            return True
        markers = (
            "thank you for watching",
            "thanks for watching",
            "subscribe",
            "like and subscribe",
        )
        for m in markers:
            if m in t:
                return True
        words = t.split()
        if len(words) >= 6:
            for i in range(len(words) - 5):
                phrase = " ".join(words[i : i + 3])
                if t.count(phrase) >= 3:
                    return True
        return False

    async def start(self):
        self._loop = asyncio.get_event_loop()
        self._running = True
        await asyncio.to_thread(self._ensure_model)
        logger.info("ASR started (Parakeet MLX %s)", self.model_id)
        return self._transcript_queue

    def stop(self):
        self._running = False
        self._speech_buffer = np.array([], dtype=np.float32)
        self._pre_buffer = np.array([], dtype=np.float32)
        logger.info("ASR stopped")
