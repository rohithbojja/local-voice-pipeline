"""
CLI Voice Pipeline — Direct mic/speaker, no browser needed.

Mic → VAD → Whisper ASR → LLM (stream) → OmniVoice TTS → Speaker

Run:  uv run cli.py
"""
import asyncio
import logging
import sys
import os
import time
import queue
import threading
import numpy as np

# Workaround for Windows PyTorch 2.6 / cuDNN Error 127
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
import torch
torch.backends.cudnn.enabled = False

import sounddevice as sd

import config
from asr_runner import ASRRunner
from llm_client import LLMClient
from tts_runner import TTSRunner
from vad import VADProcessor

# ----- Logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ["urllib3", "httpcore", "httpx", "azure", "faster_whisper", "ctranslate2"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("cli")


def preload_models():
    """Pre-load all AI models and return them."""
    print("\n  ⏳ Pre-loading AI models...")

    # 1. Whisper ASR
    print("  [1/3] Loading Whisper ASR...")
    from faster_whisper import WhisperModel
    asr_kw = {"device": config.ASR_DEVICE, "compute_type": config.ASR_COMPUTE_TYPE}
    if config.ASR_DEVICE == "cpu" and config.ASR_CPU_THREADS > 0:
        asr_kw["cpu_threads"] = config.ASR_CPU_THREADS
    asr_model = WhisperModel(config.ASR_MODEL, **asr_kw)
    print(f"  ✅ Whisper ({config.ASR_MODEL}) loaded")

    # 2. OmniVoice TTS
    print("  [2/3] Loading OmniVoice TTS...")
    tts = TTSRunner(
        model_id=config.OMNIVOICE_MODEL,
        ref_audio=config.OMNIVOICE_REF_AUDIO,
        ref_text=config.OMNIVOICE_REF_TEXT,
        device_pref=config.TTS_DEVICE,
        language=config.OMNIVOICE_LANGUAGE,
        speed=config.OMNIVOICE_SPEED,
    )
    tts._ensure_model()
    print(f"  ✅ OmniVoice ({config.OMNIVOICE_MODEL}) loaded")

    # 3. Silero VAD
    print("  [3/3] Loading Silero VAD...")
    VADProcessor(sample_rate=config.ASR_SAMPLE_RATE)
    print("  ✅ Silero VAD loaded")

    print("  🚀 All models ready!\n")
    return asr_model, tts


class CLIPipeline:
    """Full voice pipeline running in the terminal."""

    def __init__(self, asr_model, tts: TTSRunner):
        self.tts = tts
        self.llm = LLMClient(base_url=config.LLM_BASE_URL, model=config.LLM_MODEL)
        self.vad = VADProcessor(
            sample_rate=config.ASR_SAMPLE_RATE,
            threshold=0.5,
            min_speech_ms=250,
            min_silence_ms=700,
        )
        self.asr = ASRRunner(
            model_size=config.ASR_MODEL,
            device=config.ASR_DEVICE,
            compute_type=config.ASR_COMPUTE_TYPE,
            language=config.ASR_LANGUAGE,
            sample_rate=config.ASR_SAMPLE_RATE,
            cpu_threads=config.ASR_CPU_THREADS,
        )
        self.asr._model = asr_model

        self.tts_playing = False
        self.running = True

        # Audio playback queue
        self._play_q: queue.Queue[np.ndarray | None] = queue.Queue()

    def _play_audio_worker(self):
        """Background thread: plays PCM16 bytes through speakers."""
        while self.running:
            try:
                pcm = self._play_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if pcm is None:
                break
            try:
                sd.play(pcm, samplerate=self.tts.sample_rate, blocking=True)
            except Exception as e:
                logger.error(f"Playback error: {e}")

    def _status(self, msg: str):
        """Print a status line in-place."""
        sys.stdout.write(f"\r  {msg}  \r")
        sys.stdout.flush()

    async def run(self):
        loop = asyncio.get_event_loop()
        transcript_queue = await self.asr.start()

        # Start audio playback thread
        play_thread = threading.Thread(target=self._play_audio_worker, daemon=True)
        play_thread.start()

        # Start mic capture
        mic_q: queue.Queue[bytes] = queue.Queue()

        def mic_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Mic: {status}")
            mic_q.put(indata.copy().tobytes())

        stream = sd.InputStream(
            samplerate=config.ASR_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=512,
            callback=mic_callback,
        )
        stream.start()
        self._status("🎤 Listening...")
        print()

        async def process_mic():
            """Read mic chunks, run VAD, feed ASR."""
            while self.running:
                try:
                    audio_bytes = await asyncio.to_thread(mic_q.get, timeout=0.3)
                except Exception:
                    continue

                # Suppress VAD while TTS is playing
                if not self.tts_playing:
                    vad_events = self.vad.process(audio_bytes)
                else:
                    vad_events = []

                for evt in vad_events:
                    if evt["type"] == "speech_start":
                        self.asr.set_speech_active(True)
                        self._status("🎤 Hearing you...")
                    elif evt["type"] == "speech_end":
                        self.asr.set_speech_active(False)
                        self.asr.transcribe_buffer()

                self.asr.feed_audio(audio_bytes)

        async def process_transcripts():
            """Listen for ASR results, stream LLM → TTS."""
            while self.running:
                try:
                    event = await asyncio.wait_for(transcript_queue.get(), timeout=0.3)
                except asyncio.TimeoutError:
                    continue

                if event["type"] != "final":
                    continue

                transcript = event["text"]
                print(f"\n  🗣️  You: {transcript}")
                self._status("🤔 Thinking...")

                pipeline_start = time.time()
                full_response = ""
                first_sentence = True
                self.tts_playing = True

                try:
                    sentence_q: queue.Queue = queue.Queue()
                    SENTINEL = None

                    def _run_llm():
                        try:
                            for s in self.llm.stream_sentences(transcript):
                                sentence_q.put(s)
                        except Exception as exc:
                            logger.error(f"LLM error: {exc}")
                        finally:
                            sentence_q.put(SENTINEL)

                    loop.run_in_executor(None, _run_llm)

                    while True:
                        try:
                            sentence = await asyncio.to_thread(
                                sentence_q.get, timeout=0.3
                            )
                        except Exception:
                            continue

                        if sentence is SENTINEL:
                            break

                        full_response += sentence + " "

                        if first_sentence:
                            elapsed = time.time() - pipeline_start
                            logger.info(f"First sentence in {elapsed:.2f}s")
                            first_sentence = False

                        # Generate TTS
                        audio_bytes = await asyncio.to_thread(self.tts.generate, sentence)
                        if audio_bytes:
                            pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            self._play_q.put(pcm)

                except Exception as e:
                    logger.error(f"Pipeline error: {e}", exc_info=True)

                total = time.time() - pipeline_start
                self.tts_playing = False
                print(f"\n  💋 Her: {full_response.strip()}")
                logger.info(f"Pipeline total: {total:.1f}s")
                self._status("🎤 Listening...")
                print()

        # Run both loops concurrently
        try:
            await asyncio.gather(process_mic(), process_transcripts())
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            stream.stop()
            stream.close()
            self._play_q.put(None)
            play_thread.join(timeout=2)
            self.asr.stop()
            self.llm.close()
            self.tts.cleanup()
            print("\n  👋 Bye!")


def main():
    print("\n" + "=" * 60)
    print("  🎙️  CLI Voice Pipeline (No Browser)")
    print(f"  ASR: Whisper ({config.ASR_MODEL})")
    print(f"  LLM: {config.LLM_MODEL}")
    print(f"  TTS: OmniVoice ({config.OMNIVOICE_MODEL})")
    print("  Press Ctrl+C to quit")
    print("=" * 60)

    asr_model, tts = preload_models()
    pipeline = CLIPipeline(asr_model, tts)

    try:
        asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        print("\n  👋 Bye!")


if __name__ == "__main__":
    main()
