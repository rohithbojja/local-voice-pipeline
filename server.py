"""
Local Voice Pipeline — WebSocket Server (Fully Local, Open-Source)

Browser → WebSocket → VAD → Whisper ASR → LLM → F5-TTS → Audio back

Everything runs locally:
  - ASR: Faster-Whisper (GPU)
  - TTS: F5-TTS (GPU, emotional voice)
  - LLM: Ollama (local)
  - VAD: Silero (local)

All models are PRE-LOADED at server startup for instant response.

Run:  python server.py
Open: http://localhost:8890
"""
import asyncio
import logging
import base64
import json
import time
import os
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

# ===== GLOBAL MODEL SINGLETONS (preloaded at startup) =====
_asr_model = None   # WhisperModel instance (shared across sessions)
_tts_runner = None   # TTSRunner instance (shared across sessions)
_vad_model = None    # VAD model reference


def preload_all_models():
    """Load all heavy models into GPU/RAM before server starts accepting connections."""
    global _asr_model, _tts_runner, _vad_model

    print("\n  ⏳ Pre-loading AI models into GPU...")

    # 1. Load Whisper ASR model
    print("  [1/3] Loading Whisper ASR model...")
    from faster_whisper import WhisperModel
    _asr_model = WhisperModel(
        config.ASR_MODEL,
        device=config.ASR_DEVICE,
        compute_type=config.ASR_COMPUTE_TYPE,
    )
    print(f"  ✅ Whisper ({config.ASR_MODEL}) loaded on {config.ASR_DEVICE}")

    # 2. Load F5-TTS model
    print("  [2/3] Loading F5-TTS model (this takes ~15s)...")
    _tts_runner = TTSRunner(
        ref_audio_path=config.TTS_REF_AUDIO,
        ref_text=config.TTS_REF_TEXT,
        model_type=config.TTS_MODEL_TYPE,
        device=config.TTS_DEVICE,
        speed=config.TTS_SPEED,
        nfe_step=config.TTS_NFE_STEP,
    )
    _tts_runner._ensure_model()  # Force immediate load
    print(f"  ✅ F5-TTS ({config.TTS_MODEL_TYPE}) loaded on {config.TTS_DEVICE}")

    # 3. Load Silero VAD model (fast, but still preload)
    print("  [3/3] Loading Silero VAD model...")
    _vad_test = VADProcessor(sample_rate=config.ASR_SAMPLE_RATE)
    _vad_model = True  # VAD caches itself via torch.hub
    print("  ✅ Silero VAD loaded")

    print("  🚀 All models loaded! Server ready for instant responses.\n")


# ===== FastAPI App with Lifespan =====

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models before server starts accepting connections."""
    await asyncio.to_thread(preload_all_models)
    yield
    # Cleanup
    global _tts_runner
    if _tts_runner:
        _tts_runner.cleanup()
    logger.info("Server shutdown, models cleaned up")

app = FastAPI(title="Local Voice Pipeline", lifespan=lifespan)


@app.get("/")
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


class SessionState:
    """Mutable state shared across async tasks in a session."""
    def __init__(self):
        self.tts_playing = False
        self.barge_triggered = False
        self.active = True


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    # Create per-session ASR runner that uses the preloaded model
    asr = ASRRunner(
        model_size=config.ASR_MODEL,
        device=config.ASR_DEVICE,
        compute_type=config.ASR_COMPUTE_TYPE,
        language=config.ASR_LANGUAGE,
        sample_rate=config.ASR_SAMPLE_RATE,
    )
    # Inject the preloaded model instead of loading a new one
    asr._model = _asr_model

    llm = LLMClient(base_url=config.LLM_BASE_URL, model=config.LLM_MODEL)

    # Use the global preloaded TTS runner
    tts = _tts_runner

    vad = VADProcessor(sample_rate=config.ASR_SAMPLE_RATE, threshold=0.5,
                       min_speech_ms=250, min_silence_ms=700)

    state = SessionState()

    # ASR start (just sets up the queue, model already loaded)
    transcript_queue = await asr.start()

    # Tell client we're ready immediately
    await ws.send_json({"type": "status", "text": "🎤 Listening..."})

    async def send_json_safe(data: dict):
        try:
            if state.active:
                await ws.send_json(data)
        except Exception:
            pass

    async def process_transcripts():
        """Listen for ASR results, run LLM + TTS."""
        while state.active:
            try:
                event = await asyncio.wait_for(transcript_queue.get(), timeout=0.3)
            except asyncio.TimeoutError:
                continue

            transcript = event["text"]

            if event["type"] == "final":
                await send_json_safe({"type": "transcript", "text": transcript})
                logger.info(f"Transcript: {transcript}")

                # LLM
                await send_json_safe({"type": "status", "text": "🤔 Thinking..."})
                state.barge_triggered = False
                llm_start = time.time()
                response = await asyncio.to_thread(llm.send, transcript)
                llm_time = time.time() - llm_start
                await send_json_safe({"type": "response", "text": response})
                logger.info(f"LLM ({llm_time:.1f}s): {response}")

                if state.barge_triggered:
                    logger.info("⚡ Barge-in during LLM — skipping TTS")
                    await send_json_safe({"type": "status", "text": "🎤 Listening..."})
                    continue

                # TTS
                await send_json_safe({"type": "status", "text": "🔊 Generating Speech..."})
                state.tts_playing = True
                tts_start = time.time()

                if state.barge_triggered:
                    state.tts_playing = False
                    continue

                audio_bytes = await asyncio.to_thread(tts.generate, response)
                tts_time = time.time() - tts_start

                if audio_bytes and not state.barge_triggered:
                    await send_json_safe({"type": "status", "text": "🔊 Speaking..."})
                    CHUNK_SIZE = tts.sample_rate * 2  # 1 second of PCM16
                    for i in range(0, len(audio_bytes), CHUNK_SIZE):
                        if state.barge_triggered:
                            logger.info("⚡ Barge-in — stopping TTS stream")
                            break
                        chunk = audio_bytes[i:i + CHUNK_SIZE]
                        audio_b64 = base64.b64encode(chunk).decode("ascii")
                        await send_json_safe({
                            "type": "audio",
                            "data": audio_b64,
                            "sample_rate": tts.sample_rate,
                        })
                        await asyncio.sleep(0.02)
                    logger.info(f"TTS ({tts_time:.1f}s): {len(audio_bytes)} bytes")

                state.tts_playing = False
                if not state.barge_triggered:
                    await send_json_safe({"type": "status", "text": "🎤 Listening..."})

    processor_task = asyncio.create_task(process_transcripts())

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data:
                audio_bytes = data["bytes"]

                # VAD processing FIRST to set speech state
                vad_events = await asyncio.to_thread(vad.process, audio_bytes)
                for evt in vad_events:
                    if evt["type"] == "speech_start":
                        asr.set_speech_active(True)
                        await send_json_safe({"type": "vad", "speaking": True})
                        logger.info(f"VAD: speech_start | tts_playing={state.tts_playing}")
                        if state.tts_playing:
                            state.barge_triggered = True
                            logger.info("⚡ Server barge-in via VAD")
                            await send_json_safe({"type": "stop_audio"})
                    elif evt["type"] == "speech_end":
                        asr.set_speech_active(False)
                        await send_json_safe({"type": "vad", "speaking": False})
                        logger.info("VAD: speech_end — triggering Whisper transcription")
                        await asyncio.to_thread(asr.transcribe_buffer)

                # Feed audio to ASR buffer (only accumulates during active speech)
                asr.feed_audio(audio_bytes)

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "reset":
                    llm.reset()
                    vad.reset()
                    await send_json_safe({"type": "status", "text": "🔄 Conversation reset"})
                elif msg.get("type") == "barge_in":
                    state.barge_triggered = True
                    state.tts_playing = False
                    logger.info("⚡ Client barge-in received")

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        state.active = False
        processor_task.cancel()
        asr.stop()
        llm.close()
        # Don't cleanup TTS — it's a global singleton
        logger.info("Session cleaned up")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🎙️  Local Voice Pipeline (100% Local)")
    print(f"  ASR: Whisper ({config.ASR_MODEL})  |  LLM: {config.LLM_MODEL}")
    print(f"  TTS: F5-TTS ({config.TTS_MODEL_TYPE})  |  VAD: Silero")
    print(f"  Speed: {config.TTS_SPEED}  |  NFE Steps: {config.TTS_NFE_STEP}")
    print(f"  Open: http://localhost:{config.SERVER_PORT}")
    print("=" * 60)
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, log_level="info")
