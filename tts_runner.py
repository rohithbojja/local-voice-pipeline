"""
TTS Runner — F5-TTS (local GPU inference, emotional voice synthesis).
Zero-shot voice cloning with expressive, emotional output.

Based on official F5-TTS API: https://github.com/SWivid/F5-TTS
"""
import logging
import numpy as np
import torch
import asyncio

logger = logging.getLogger(__name__)


class TTSRunner:
    """
    F5-TTS — local GPU TTS with emotional, expressive voice synthesis.
    
    Important notes from F5-TTS docs:
    - Reference audio should be < 12 seconds with ~1s silence at end
    - The model clones both voice AND speaking rate from reference
    - 'speed' param controls text chunking (lower = slower speech)
    - Total generation is max 30s (including ref audio)
    """

    def __init__(self, ref_audio_path: str = "", ref_text: str = "",
                 model_type: str = "F5TTS_v1_Base", device: str = None,
                 speed: float = 0.8, nfe_step: int = 32):
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.speed = speed  # Lower = fewer chars per chunk = slower speech
        self.nfe_step = nfe_step
        self._api = None
        self.sample_rate = 24000  # F5-TTS outputs 24kHz

    def _ensure_model(self):
        if self._api is None:
            logger.info(f"Loading F5-TTS model ({self.model_type}) on {self.device}...")
            from f5_tts.api import F5TTS
            self._api = F5TTS(model=self.model_type, device=self.device)
            logger.info("F5-TTS model loaded into VRAM")

    async def start(self):
        """Preload the model in the background."""
        await asyncio.to_thread(self._ensure_model)

    def generate(self, text: str) -> bytes:
        """Generate PCM16 audio bytes from text."""
        self._ensure_model()
        try:
            # Clean text: strip emojis, special chars, add pauses
            gen_text = self._clean_text_for_tts(text)
            
            logger.info(f"TTS generating (speed={self.speed}, nfe={self.nfe_step}): {gen_text[:80]}...")

            wav, sr, _ = self._api.infer(
                ref_file=self.ref_audio_path if self.ref_audio_path else "",
                ref_text=self.ref_text if self.ref_text else "",
                gen_text=gen_text,
                speed=self.speed,
                nfe_step=self.nfe_step,
                file_wave=None,
                seed=None,
            )
            self.sample_rate = sr

            # Convert to PCM16 bytes
            if isinstance(wav, torch.Tensor):
                wav = wav.cpu().numpy()
            if wav.ndim > 1:
                wav = wav.squeeze()

            # Normalize to [-1, 1] range
            max_val = max(abs(wav.max()), abs(wav.min()))
            if max_val > 0:
                wav = wav / max_val * 0.95  # Leave headroom

            pcm16 = (wav * 32767).astype(np.int16)
            audio_bytes = pcm16.tobytes()
            duration = len(pcm16) / sr
            logger.info(f"TTS generated {len(audio_bytes)} bytes ({duration:.1f}s) at {sr}Hz")
            return audio_bytes

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            return b""

    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS: strip emojis, special chars, add natural pauses."""
        import re
        
        # Remove emojis and special Unicode characters (F5-TTS can't pronounce them)
        # This regex removes most emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "\U0001f926-\U0001f937"  # additional emoticons
            "\U00010000-\U0010ffff"  # supplementary chars
            "\u200d"                 # zero width joiner
            "\u2640-\u2642"          # gender symbols
            "\ufe0f"                 # variation selector
            "\u2600-\u26FF"          # misc symbols
            "\u2700-\u27BF"          # dingbats
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub("", text)
        
        # Remove asterisks, markdown formatting
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'_+', ' ', text)
        text = re.sub(r'#+\s*', '', text)
        
        # Remove parenthetical stage directions like (laughs) (sighs) etc from LLM
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # F5-TTS docs: "Add spaces or punctuation to introduce pauses"
        text = text.replace("! ", "!  ")
        text = text.replace("? ", "?  ")
        text = text.replace(". ", ".  ")
        text = text.replace(", ", ",  ")
        
        return text

    def cleanup(self):
        if self._api is not None:
            del self._api
            self._api = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
