"""
Local Voice Pipeline — WebSocket Server with VAD, Barge-in, Streaming TTS

Browser sends mic audio over WebSocket → VAD → ASR → LLM → TTS (streaming) → audio back.

Features:
  - Voice Activity Detection (Silero VAD) — detects speech start/stop
  - Barge-in — client + server side: interrupts TTS when user speaks
  - Streaming TTS — sends audio chunks as they're generated

Run:  python server.py
Open: http://localhost:8890
"""
import asyncio
import logging
import base64
import json
import time
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path

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
for noisy in ["urllib3", "httpcore", "httpx", "azure", "websockets", "uvicorn.access"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger("pipeline")

app = FastAPI(title="Local Voice Pipeline")


@app.get("/")
async def index():
    """Serve the browser client."""
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

    # ----- Per-connection instances -----
    asr = ASRRunner(
        key=config.MICROSOFT_KEY,
        region=config.MICROSOFT_REGION,
        language=config.ASR_LANGUAGE,
        sample_rate=config.ASR_SAMPLE_RATE,
    )
    llm = LLMClient(base_url=config.LLM_BASE_URL, model=config.LLM_MODEL)
    tts = TTSRunner(
        api_key=config.ELEVENLABS_API_KEY,
        base_url=config.ELEVENLABS_BASE_URL,
        voice_id=config.TTS_VOICE,
        model_id=config.TTS_MODEL,
    )
    vad = VADProcessor(sample_rate=config.ASR_SAMPLE_RATE, threshold=0.5,
                       min_speech_ms=250, min_silence_ms=700)

    state = SessionState()
    transcript_queue = await asr.start()

    async def send_json_safe(data: dict):
        """Send JSON to client, ignore if disconnected."""
        try:
            if state.active:
                await ws.send_json(data)
        except Exception:
            pass

    async def process_transcripts():
        """Listen for ASR results, run LLM + TTS with barge-in support."""
        while state.active:
            try:
                event = await asyncio.wait_for(transcript_queue.get(), timeout=0.3)
            except asyncio.TimeoutError:
                continue

            if event["type"] == "interim":
                await send_json_safe({"type": "interim", "text": event["text"]})
                continue

            if event["type"] == "final":
                transcript = event["text"]
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

                # Check if user already barged in during LLM processing
                if state.barge_triggered:
                    logger.info("⚡ Barge-in during LLM — skipping TTS")
                    await send_json_safe({"type": "status", "text": "🎤 Listening..."})
                    continue

                # TTS — generate and stream
                await send_json_safe({"type": "status", "text": "🔊 Speaking..."})
                state.tts_playing = True

                tts_start = time.time()
                audio_bytes = await asyncio.to_thread(tts.generate, response)
                tts_time = time.time() - tts_start

                if audio_bytes and not state.barge_triggered:
                    # Send audio in chunks for streaming playback
                    CHUNK_SIZE = 24000 * 2  # 1 second of PCM16 at 24kHz
                    for i in range(0, len(audio_bytes), CHUNK_SIZE):
                        if state.barge_triggered:
                            logger.info(f"⚡ Barge-in at chunk {i // CHUNK_SIZE} — stopping TTS stream")
                            break
                        chunk = audio_bytes[i:i + CHUNK_SIZE]
                        audio_b64 = base64.b64encode(chunk).decode("ascii")
                        await send_json_safe({
                            "type": "audio",
                            "data": audio_b64,
                            "sample_rate": config.TTS_SAMPLE_RATE,
                        })
                        # Yield to allow barge-in events to be processed
                        await asyncio.sleep(0.02)

                    logger.info(f"TTS ({tts_time:.1f}s): sent {len(audio_bytes)} bytes")
                elif state.barge_triggered:
                    logger.info("⚡ Barge-in during TTS generation — audio discarded")

                state.tts_playing = False
                if not state.barge_triggered:
                    await send_json_safe({"type": "status", "text": "🎤 Listening..."})

    # Start transcript processor
    processor_task = asyncio.create_task(process_transcripts())

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data:
                audio_bytes = data["bytes"]

                # VAD: detect speech activity
                vad_events = await asyncio.to_thread(vad.process, audio_bytes)
                for evt in vad_events:
                    if evt["type"] == "speech_start":
                        await send_json_safe({"type": "vad", "speaking": True})
                        logger.info(f"VAD speech_start | tts_playing={state.tts_playing}")
                        # Server-side barge-in
                        if state.tts_playing:
                            state.barge_triggered = True
                            logger.info("⚡ Server barge-in triggered via VAD")
                            await send_json_safe({"type": "stop_audio"})
                    elif evt["type"] == "speech_end":
                        await send_json_safe({"type": "vad", "speaking": False})

                # Feed audio to ASR
                asr.feed_audio(audio_bytes)

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "reset":
                    llm.reset()
                    vad.reset()
                    await send_json_safe({"type": "status", "text": "🔄 Conversation reset"})
                elif msg.get("type") == "barge_in":
                    # Client detected barge-in
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
        tts.cleanup()
        logger.info("Session cleaned up")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🎙️  Local Voice Pipeline (WebSocket)")
    print(f"  ASR: Microsoft  |  LLM: {config.LLM_MODEL}")
    print(f"  TTS: ElevenLabs  |  VAD: Silero  |  Barge-in: ON")
    print(f"  Open: http://localhost:{config.SERVER_PORT}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, log_level="info")
