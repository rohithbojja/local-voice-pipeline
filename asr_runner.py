"""
ASR Runner — Standalone Azure Speech SDK wrapper.
No orchestrator imports. Feeds audio chunks → yields transcripts.
"""
import asyncio
import queue
import logging
import azure.cognitiveservices.speech as speechsdk

logger = logging.getLogger(__name__)


class ASRRunner:
    """Azure Speech SDK continuous recognition — standalone, no multiprocessing."""

    def __init__(self, key: str, region: str, language: str = "en-IN", sample_rate: int = 16000):
        self.key = key
        self.region = region
        self.language = language
        self.sample_rate = sample_rate
        self.recognizer = None
        self.push_stream = None
        self._transcript_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._running = False
        self._loop = None

    def _setup(self):
        speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        speech_config.speech_recognition_language = self.language
        speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1500")
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "15000")
        speech_config.set_property(speechsdk.PropertyId.SpeechServiceResponse_StablePartialResultThreshold, "1")
        speech_config.set_profanity(speechsdk.ProfanityOption.Raw)

        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=self.sample_rate, bits_per_sample=16, channels=1
        )
        self.push_stream = speechsdk.audio.PushAudioInputStream(stream_format=audio_format)
        audio_config = speechsdk.audio.AudioConfig(stream=self.push_stream)
        self.recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        self.recognizer.recognized.connect(self._on_recognized)
        self.recognizer.recognizing.connect(self._on_recognizing)
        self.recognizer.canceled.connect(self._on_canceled)

    def _on_recognized(self, evt):
        text = evt.result.text.strip() if evt.result.text else ""
        if text:
            logger.info(f"ASR Final: {text}")
            try:
                self._loop.call_soon_threadsafe(
                    self._transcript_queue.put_nowait, {"type": "final", "text": text}
                )
            except Exception as e:
                logger.error(f"Error queuing transcript: {e}")

    def _on_recognizing(self, evt):
        text = evt.result.text.strip() if evt.result.text else ""
        if text:
            try:
                self._loop.call_soon_threadsafe(
                    self._transcript_queue.put_nowait, {"type": "interim", "text": text}
                )
            except Exception:
                pass

    def _on_canceled(self, evt):
        details = evt.result.cancellation_details
        if details.reason == speechsdk.CancellationReason.Error:
            logger.error(f"ASR Canceled: {details.error_details}")

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM16 audio bytes into the recognizer."""
        if self.push_stream and self._running:
            self.push_stream.write(audio_bytes)

    async def start(self):
        """Start continuous recognition. Returns async queue of transcript events."""
        self._loop = asyncio.get_event_loop()
        self._running = True
        self._setup()
        self.recognizer.start_continuous_recognition()
        logger.info(f"ASR started (lang={self.language}, rate={self.sample_rate}Hz)")
        return self._transcript_queue

    def stop(self):
        self._running = False
        try:
            if self.recognizer:
                self.recognizer.stop_continuous_recognition()
                self.recognizer.recognized.disconnect_all()
                self.recognizer.recognizing.disconnect_all()
                self.recognizer.canceled.disconnect_all()
            if self.push_stream:
                self.push_stream.close()
        except Exception as e:
            logger.warning(f"ASR cleanup: {e}")
        logger.info("ASR stopped")
