"""
Local Voice Pipeline — Ultra-Fast WebSocket Server

Browser → WebSocket → VAD → Whisper ASR → LLM (stream) → OmniVoice TTS → Audio back

All models pre-loaded at startup. LLM streams sentence-by-sentence,
TTS generates per-sentence for minimal latency.

Run:  python server.py
Open: http://localhost:8890
"""
import asyncio
import logging
import base64
import json
import time
import os
import queue as _queue
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path
from contextlib import asynccontextmanager

# Workaround for Windows PyTorch 2.6 / cuDNN Error 127
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
import torch
torch.backends.cudnn.enabled = False

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
for noisy in ["urllib3", "httpcore", "httpx", "azure", "websockets", "uvicorn.access",
              "faster_whisper", "ctranslate2"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("pipeline")

# ===== GLOBAL SINGLETONS =====
_asr_model = None
_tts_runner = None


def preload_all_models():
    global _asr_model, _tts_runner

    print("\n  ⏳ Pre-loading AI models...")

    # 1. Whisper ASR
    print("  [1/3] Loading Whisper ASR...")
    from faster_whisper import WhisperModel

    _asr_kw: dict = {
        "device": config.ASR_DEVICE,
        "compute_type": config.ASR_COMPUTE_TYPE,
    }
    if config.ASR_DEVICE == "cpu" and config.ASR_CPU_THREADS > 0:
        _asr_kw["cpu_threads"] = config.ASR_CPU_THREADS
    _asr_model = WhisperModel(config.ASR_MODEL, **_asr_kw)
    print(f"  ✅ Whisper ({config.ASR_MODEL}) loaded")

    # 2. OmniVoice TTS
    print("  [2/3] Loading OmniVoice TTS...")
    _tts_runner = TTSRunner(
        model_id=config.OMNIVOICE_MODEL,
        ref_audio=config.OMNIVOICE_REF_AUDIO,
        ref_text=config.OMNIVOICE_REF_TEXT,
        device_pref=config.TTS_DEVICE,
        language=config.OMNIVOICE_LANGUAGE,
        speed=config.OMNIVOICE_SPEED,
    )
    _tts_runner._ensure_model()
    print(f"  ✅ OmniVoice ({config.OMNIVOICE_MODEL}) + clone ref loaded")

    # 3. Silero VAD
    print("  [3/3] Loading Silero VAD...")
    VADProcessor(sample_rate=config.ASR_SAMPLE_RATE)
    print("  ✅ Silero VAD loaded")

    print("  🚀 All models ready! Ultra-fast pipeline active.\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(preload_all_models)
    yield
    if _tts_runner:
        _tts_runner.cleanup()

app = FastAPI(title="Local Voice Pipeline", lifespan=lifespan)


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


class SessionState:
    def __init__(self):
        self.tts_playing = False
        self.barge_triggered = False
        self.active = True


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    asr = ASRRunner(
        model_size=config.ASR_MODEL,
        device=config.ASR_DEVICE,
        compute_type=config.ASR_COMPUTE_TYPE,
        language=config.ASR_LANGUAGE,
        sample_rate=config.ASR_SAMPLE_RATE,
        cpu_threads=config.ASR_CPU_THREADS,
    )
    asr._model = _asr_model  # Use preloaded model

    llm = LLMClient(base_url=config.LLM_BASE_URL, model=config.LLM_MODEL)
    tts = _tts_runner
    vad = VADProcessor(sample_rate=config.ASR_SAMPLE_RATE, threshold=0.5,
                       min_speech_ms=250, min_silence_ms=700)

    state = SessionState()
    transcript_queue = await asr.start()
    await ws.send_json({"type": "status", "text": "🎤 Listening..."})

    async def send_json_safe(data: dict):
        try:
            if state.active:
                await ws.send_json(data)
        except Exception:
            pass

    async def send_audio_chunk(audio_bytes: bytes):
        """Send a chunk of audio to the browser."""
        CHUNK_SIZE = tts.sample_rate * 2  # 1 second of PCM16
        for i in range(0, len(audio_bytes), CHUNK_SIZE):
            if state.barge_triggered:
                logger.info("⚡ Barge-in — stopping TTS stream")
                return
            chunk = audio_bytes[i:i + CHUNK_SIZE]
            audio_b64 = base64.b64encode(chunk).decode("ascii")
            await send_json_safe({
                "type": "audio",
                "data": audio_b64,
                "sample_rate": tts.sample_rate,
            })
            await asyncio.sleep(0.01)

    async def process_transcripts():
        """Listen for ASR results, stream LLM → TTS."""
        while state.active:
            try:
                event = await asyncio.wait_for(transcript_queue.get(), timeout=0.3)
            except asyncio.TimeoutError:
                continue

            if event["type"] != "final":
                continue

            transcript = event["text"]
            await send_json_safe({"type": "transcript", "text": transcript})
            logger.info(f"Transcript: {transcript}")

            # Stream LLM → TTS sentence by sentence
            await send_json_safe({"type": "status", "text": "🤔 Thinking..."})
            state.barge_triggered = False
            state.tts_playing = True

            pipeline_start = time.time()
            full_response = ""
            first_audio = True

            try:
                # Use a thread-safe queue so the LLM thread can push
                # sentences one at a time while we consume them async
                sentence_q: _queue.Queue = _queue.Queue()
                SENTINEL = None  # marks end of stream

                def _run_llm():
                    try:
                        for s in llm.stream_sentences(transcript):
                            sentence_q.put(s)
                    except Exception as exc:
                        logger.error(f"LLM thread error: {exc}", exc_info=True)
                    finally:
                        sentence_q.put(SENTINEL)

                llm_task = asyncio.get_event_loop().run_in_executor(None, _run_llm)

                while True:
                    if state.barge_triggered:
                        logger.info("⚡ Barge-in during pipeline")
                        break
                    try:
                        sentence = await asyncio.to_thread(
                            sentence_q.get, timeout=0.3
                        )
                    except Exception:
                        # queue.get timed out, loop and re-check barge
                        continue

                    if sentence is SENTINEL:
                        break  # LLM finished

                    full_response += sentence + " "

                    if first_audio:
                        elapsed = time.time() - pipeline_start
                        logger.info(f"LLM first sentence in {elapsed:.2f}s")
                        await send_json_safe({"type": "status", "text": "🔊 Speaking..."})
                        first_audio = False

                    # Generate TTS for this sentence
                    tts_start = time.time()
                    audio_bytes = await asyncio.to_thread(tts.generate, sentence)
                    tts_time = time.time() - tts_start
                    logger.info(f"TTS sentence ({tts_time:.2f}s): {sentence[:60]}")

                    if audio_bytes and not state.barge_triggered:
                        await send_audio_chunk(audio_bytes)

                # Wait for LLM thread to finish cleanly
                await llm_task

            except Exception as e:
                logger.error(f"Pipeline error: {e}", exc_info=True)

            total_time = time.time() - pipeline_start
            logger.info(f"Pipeline total: {total_time:.1f}s | Response: {full_response.strip()[:80]}")

            await send_json_safe({"type": "response", "text": full_response.strip()})
            state.tts_playing = False

            # Drain any transcripts that queued during TTS
            while not transcript_queue.empty():
                try:
                    transcript_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            if not state.barge_triggered:
                await send_json_safe({"type": "status", "text": "🎤 Listening..."})

    processor_task = asyncio.create_task(process_transcripts())

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data:
                audio_bytes = data["bytes"]

                # Suppress VAD while TTS is playing to avoid
                # the mic picking up speaker audio as "speech"
                if not state.tts_playing:
                    vad_events = await asyncio.to_thread(vad.process, audio_bytes)
                else:
                    vad_events = []

                for evt in vad_events:
                    if evt["type"] == "speech_start":
                        asr.set_speech_active(True)
                        await send_json_safe({"type": "vad", "speaking": True})
                        if state.tts_playing:
                            state.barge_triggered = True
                            logger.info("⚡ Server barge-in via VAD")
                            await send_json_safe({"type": "stop_audio"})
                    elif evt["type"] == "speech_end":
                        asr.set_speech_active(False)
                        await send_json_safe({"type": "vad", "speaking": False})
                        logger.info("VAD: speech_end → transcribing")
                        await asyncio.to_thread(asr.transcribe_buffer)

                asr.feed_audio(audio_bytes)

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "reset":
                    llm.reset()
                    vad.reset()
                    await send_json_safe({"type": "status", "text": "🔄 Reset"})
                elif msg.get("type") == "barge_in":
                    state.barge_triggered = True
                    state.tts_playing = False

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        state.active = False
        processor_task.cancel()
        asr.stop()
        llm.close()
        logger.info("Session cleaned up")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🎙️  Ultra-Fast Local Voice Pipeline")
    print(f"  ASR: Whisper ({config.ASR_MODEL})")
    print(f"  LLM: {config.LLM_MODEL} (streaming)")
    print(f"  TTS: OmniVoice ({config.OMNIVOICE_MODEL})")
    print(f"  Open: http://localhost:{config.SERVER_PORT}")
    print("=" * 60)
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, log_level="info")
