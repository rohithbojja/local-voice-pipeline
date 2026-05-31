"""
TTS — Supertonic (supertone-inc/supertonic): lightning-fast, on-device ONNX TTS.
https://github.com/supertone-inc/supertonic
Install: pip install 'supertonic[serve]'

Expression tags (Supertonic 3):
    <laugh> <breath> <sigh> <cough> <sneeze> <groan> <hmm> <uh> <um> <yawn>

OmniVoice [bracket-style] tags emitted by the LLM are mapped → stripped/translated.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OmniVoice → Supertonic expression-tag translation table.
# Tags not listed here are stripped entirely.
# ---------------------------------------------------------------------------
_OMNIVOICE_TO_SUPERTONIC: dict[str, str] = {
    "[laughter]":        "<laugh>",
    "[sigh]":            "<sigh>",
    "[breath]":          "<breath>",
    "[cough]":           "<cough>",
    "[sneeze]":          "<sneeze>",
    "[groan]":           "<groan>",
    "[hmm]":             "<hmm>",
    "[uh]":              "<uh>",
    "[um]":              "<um>",
    "[yawn]":            "<yawn>",
    # These OmniVoice tags have no direct Supertonic equivalent → strip
    "[confirmation-en]": "",
    "[question-en]":     "",
    "[question-ah]":     "",
    "[question-oh]":     "",
    "[question-ei]":     "",
    "[question-yi]":     "",
    "[surprise-ah]":     "",
    "[surprise-oh]":     "",
    "[surprise-wa]":     "",
    "[surprise-yo]":     "",
    "[dissatisfaction-hnn]": "",
}

# Valid Supertonic expression tags (complete list from v3 docs)
_SUPERTONIC_TAGS: frozenset[str] = frozenset({
    "<laugh>", "<breath>", "<sigh>", "<cough>",
    "<sneeze>", "<groan>", "<hmm>", "<uh>", "<um>", "<yawn>",
})

# Pre-compiled regex for OmniVoice [bracket] tags
_OMNIVOICE_TAG_RE = re.compile(
    r"\[(?:laughter|sigh|breath|cough|sneeze|groan|hmm|uh|um|yawn"
    r"|confirmation-en|question-(?:en|ah|oh|ei|yi)"
    r"|surprise-(?:ah|oh|wa|yo)"
    r"|dissatisfaction-hnn)\]"
)

# Emoji / markdown cleanup
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0\U000024C2-\U0001F251"
    "\U00010000-\U0010ffff\u200d\ufe0f"
    "\u2600-\u26FF\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)


class TTSRunner:
    """Supertonic inference — preset voices or Voice Builder JSON for voice cloning."""

    def __init__(
        self,
        voice: str = "F1",
        voice_json: Optional[str] = None,
        lang: Optional[str] = "en",
        speed: float = 1.05,
        total_steps: int = 5,          # 5 = balanced speed/quality (docs: 2=fast,5=balanced,10=high)
        silence_duration: float = 0.15, # shorter pauses between chunks for real-time feel
        intra_op_threads: Optional[int] = None,
        inter_op_threads: Optional[int] = None,
    ):
        self.voice = voice
        self.voice_json = voice_json
        self.lang = lang
        self.speed = speed
        self.total_steps = total_steps
        self.silence_duration = silence_duration
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads

        # Supertonic outputs 44100 Hz
        self.sample_rate = 44100
        self._tts = None
        self._style = None

    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._tts is not None:
            return

        from supertonic import TTS

        label = f"json={self.voice_json}" if self.voice_json else f"preset={self.voice}"
        logger.info("Loading Supertonic (%s, lang=%s, steps=%d)…",
                    label, self.lang, self.total_steps)

        self._tts = TTS(
            auto_download=True,
            intra_op_num_threads=self.intra_op_threads,
            inter_op_num_threads=self.inter_op_threads,
        )

        if self.voice_json:
            json_path = Path(self.voice_json).expanduser()
            if not json_path.is_file():
                raise FileNotFoundError(
                    f"Voice Builder JSON not found: {json_path.resolve()}. "
                    "Download from https://supertonic.supertone.ai/voice-builder "
                    "and set SUPERTONIC_VOICE_JSON in .env"
                )
            self._style = self._tts.get_voice_style_from_path(json_path)
            logger.info("Supertonic: loaded cloned voice from %s", json_path)
        else:
            self._style = self._tts.get_voice_style(voice_name=self.voice)
            logger.info("Supertonic: using built-in preset '%s'", self.voice)

        sr = getattr(self._tts, "sample_rate", None)
        if sr:
            self.sample_rate = int(sr)
        logger.info("Supertonic ready (sample_rate=%s)", self.sample_rate)

    # ------------------------------------------------------------------
    def generate(self, text: str) -> bytes:
        """Return PCM-16 mono bytes at ``self.sample_rate``."""
        self._ensure_model()
        text = self._clean_text(text)
        if not text:
            return b""

        try:
            wav, duration = self._tts.synthesize(
                text=text,
                voice_style=self._style,
                lang=self.lang,
                total_steps=self.total_steps,
                speed=self.speed,
                silence_duration=self.silence_duration,
            )
            # wav: float32 ndarray (1, num_samples); duration: scalar or (1,) array
            audio = wav.squeeze()

            # Trim to actual duration
            dur_val = float(duration) if not hasattr(duration, "__len__") else float(duration[0])
            dur_samples = int(self.sample_rate * dur_val)
            audio = audio[:dur_samples]

            # Peak-normalise to 0.95
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 0:
                audio = audio / peak * 0.95

            pcm16 = (audio * 32767.0).astype(np.int16)
            audio_bytes = pcm16.tobytes()
            logger.info(
                "Supertonic → %d bytes (%.2fs)",
                len(audio_bytes),
                len(pcm16) / self.sample_rate,
            )
            return audio_bytes

        except Exception as e:
            logger.error("Supertonic error: %s", e, exc_info=True)
            return b""

    # ------------------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        """
        1. Translate OmniVoice [bracket] tags → Supertonic <angle> equivalents.
        2. Strip emoji, markdown artifacts.
        3. Strip any remaining <angle> tags not in the valid Supertonic set.
        """
        # 1. Translate / strip OmniVoice bracket tags
        def _translate_omnivoice(m: re.Match) -> str:
            return _OMNIVOICE_TO_SUPERTONIC.get(m.group(0), "")

        text = _OMNIVOICE_TAG_RE.sub(_translate_omnivoice, text)

        # Also catch any stray [anything-else] bracket tags the LLM invented
        text = re.sub(r"\[[^\]]+\]", "", text)

        # 2. Emoji + markdown cleanup
        text = _EMOJI_RE.sub("", text)
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"_+", " ", text)
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # 3. Strip <angle> tags not in the valid Supertonic set
        def _strip_invalid_angle(m: re.Match) -> str:
            return m.group(0) if m.group(0) in _SUPERTONIC_TAGS else ""

        text = re.sub(r"<[^>]+>", _strip_invalid_angle, text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ------------------------------------------------------------------
    def cleanup(self) -> None:
        self._style = None
        if self._tts is not None:
            del self._tts
            self._tts = None
