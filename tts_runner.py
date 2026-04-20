"""
TTS — OmniVoice (k2-fsa/OmniVoice): always uses a fixed reference clip for zero-shot cloning.
https://huggingface.co/k2-fsa/OmniVoice
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _device_map_and_dtype(device_pref: str) -> tuple[str, torch.dtype]:
    """Pick Hugging Face device_map and dtype from a coarse preference (cuda / mps / cpu)."""
    p = (device_pref or "cuda").lower().strip()
    if p.startswith("cuda"):
        if torch.cuda.is_available():
            return ("cuda:0" if p in ("cuda", "cuda:0") else p), torch.float16
        logger.warning("CUDA requested but not available; using CPU.")
    if p.startswith("mps"):
        if torch.backends.mps.is_available():
            return ("mps:0" if p == "mps" else p), torch.float16
        logger.warning("MPS requested but not available; using CPU.")
    return "cpu", torch.float32


class TTSRunner:
    """OmniVoice inference with a single cached voice-clone prompt (reference WAV + transcript)."""

    def __init__(
        self,
        model_id: str,
        ref_audio: str,
        ref_text: str,
        device_pref: str,
        language: Optional[str],
        speed: Optional[float],
        generation_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.device_pref = device_pref
        self.language = language
        self.speed = speed
        self.generation_kwargs = generation_kwargs or {}
        self._model = None
        self._clone_prompt = None
        self.sample_rate = 24000

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        from omnivoice import OmniVoice

        device_map, torch_dtype = _device_map_and_dtype(self.device_pref)
        logger.info("Loading OmniVoice %s on %s (%s)...", self.model_id, device_map, torch_dtype)

        self._model = OmniVoice.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        sr = getattr(self._model, "sampling_rate", None)
        if sr:
            self.sample_rate = int(sr)

        path = Path(self.ref_audio).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Reference audio not found: {path.resolve()}. "
                "Add reference.wav beside config.py or set OMNIVOICE_REF_AUDIO."
            )
        transcript = (self.ref_text or "").strip()
        if not transcript:
            raise ValueError(
                "Reference transcript is empty. Set OMNIVOICE_REF_TEXT or add reference.txt "
                "(exact words spoken in the reference WAV) beside config.py."
            )
        self._clone_prompt = self._model.create_voice_clone_prompt(
            str(path.resolve()),
            transcript,
        )
        logger.info("OmniVoice voice-clone prompt built from %s", path)

        logger.info("OmniVoice loaded (sample_rate=%s)", self.sample_rate)

    def generate(self, text: str) -> bytes:
        """PCM16 mono bytes at ``self.sample_rate``."""
        self._ensure_model()
        text = self._clean_text(text)
        if not text:
            return b""

        try:
            kwargs = dict(self.generation_kwargs)
            kwargs["voice_clone_prompt"] = self._clone_prompt
            if self.language:
                kwargs["language"] = self.language
            if self.speed is not None:
                kwargs["speed"] = self.speed

            audios = self._model.generate(text=text, **kwargs)
            if not audios:
                logger.warning("OmniVoice produced no audio")
                return b""

            wav = np.concatenate([np.asarray(a, dtype=np.float32) for a in audios])
            peak = float(np.max(np.abs(wav))) if wav.size else 0.0
            if peak > 0:
                wav = wav / peak * 0.95

            pcm16 = (wav * 32767.0).astype(np.int16)
            audio_bytes = pcm16.tobytes()
            logger.info(
                "OmniVoice generated %d bytes (%.2fs)",
                len(audio_bytes),
                len(pcm16) / self.sample_rate,
            )
            return audio_bytes
        except Exception as e:
            logger.error("OmniVoice error: %s", e, exc_info=True)
            return b""

    def _clean_text(self, text: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0\U000024C2-\U0001F251"
            "\U00010000-\U0010ffff\u200d\ufe0f"
            "\u2600-\u26FF\u2700-\u27BF"
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub("", text)
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"_+", " ", text)
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def cleanup(self) -> None:
        self._clone_prompt = None
        if self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
