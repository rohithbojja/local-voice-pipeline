"""
LLM Client — Ollama-compatible chat client with streaming support.
Streams responses sentence-by-sentence for ultra-fast TTS pipelining.
"""
import json
import logging
import time
import httpx
import re

logger = logging.getLogger(__name__)

# Split streaming text into TTS-sized chunks. Include full-width / CJK sentence
# marks so Japanese (e.g. 。) does not buffer the entire reply into one OmniVoice call.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s*")
# If the model never hits sentence punctuation, still chunk TTS (word boundary).
_MAX_CHUNK_CHARS = 220

SYSTEM_PROMPT = (
    "You are a sweet, cheerful, and caring female voice assistant with a warm, anime-inspired personality. "
    "Speak in a cute, friendly, and slightly playful tone. Keep your responses concise and natural "
    "for spoken conversation — ideally 1-3 sentences. "
    "Answer directly with what should be spoken aloud only — no chain-of-thought, no reasoning blocks, "
    "no tags like think or redacted. "
    "IMPORTANT: Do NOT use emojis, emoticons, special characters, asterisks, or any formatting. "
    "Your output goes directly to a text-to-speech engine that can ONLY read plain English text. "
    "Use words like 'hee hee' or 'aww' instead of emojis to express emotions. "
    "Be enthusiastic, supportive, and speak naturally with proper punctuation."
)


class LLMClient:
    """Ollama chat client with both blocking and streaming modes."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "gemma3:4b",
        reasoning_effort: str | None = "none",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = (reasoning_effort or "").strip() or None
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Long gaps between stream chunks are normal for big local models; avoid read timeouts.
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0)
        )
        logger.info(
            "LLM initialized (url=%s, model=%s, reasoning_effort=%s)",
            self.base_url,
            self.model,
            self.reasoning_effort or "(omitted)",
        )

    def _ollama_chat_options(self) -> dict:
        """Extra fields for Ollama OpenAI-compatible /v1/chat/completions."""
        if not self.reasoning_effort:
            return {}
        return {"reasoning_effort": self.reasoning_effort}

    def send(self, text: str) -> str:
        """Send user text and return full response (blocking)."""
        try:
            self.history.append({"role": "user", "content": text})
            body = {
                "model": self.model,
                "messages": self.history,
                "max_tokens": 256,
                "temperature": 0.7,
                **self._ollama_chat_options(),
            }
            resp = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"].strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I could not process that."

    def _maybe_flush_oversized_buffer(self, buffer: str):
        """If buffer grows huge without punctuation, yield a prefix at a word boundary."""
        if len(buffer) < _MAX_CHUNK_CHARS:
            return None, buffer
        cut = buffer.rfind(" ", 0, _MAX_CHUNK_CHARS)
        if cut < 20:
            cut = _MAX_CHUNK_CHARS
        chunk = buffer[:cut].strip()
        rest = buffer[cut:].lstrip()
        if not chunk:
            return None, buffer
        return chunk, rest

    def stream_sentences(self, text: str):
        """Stream LLM response, yielding complete sentences as they form."""
        try:
            self.history.append({"role": "user", "content": text})
            buffer = ""
            full_response = ""
            t0 = time.perf_counter()
            first_logged = False

            logger.info("LLM stream request started (model=%s)", self.model)
            body = {
                "model": self.model,
                "messages": self.history,
                "max_tokens": 256,
                "temperature": 0.7,
                "stream": True,
                **self._ollama_chat_options(),
            }
            with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=body,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        err = chunk.get("error")
                        if err:
                            logger.error("LLM stream API error: %s", err)
                            break
                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            if not first_logged:
                                first_logged = True
                                logger.info(
                                    "LLM first token after %.1fs",
                                    time.perf_counter() - t0,
                                )
                            buffer += token
                            full_response += token

                            sentences = _SENTENCE_BOUNDARY.split(buffer, maxsplit=1)
                            if len(sentences) > 1:
                                yield sentences[0]
                                buffer = sentences[1]
                            else:
                                flushed, buffer = self._maybe_flush_oversized_buffer(buffer)
                                if flushed is not None:
                                    yield flushed
                    except json.JSONDecodeError as e:
                        logger.warning("LLM stream bad JSON (%s): %s...", e, data[:120])
                    except (KeyError, IndexError, TypeError) as e:
                        logger.warning("LLM stream unexpected chunk (%s): %s...", e, data[:120])

            # Yield remaining buffer
            if buffer.strip():
                yield buffer.strip()

            logger.info(
                "LLM stream finished in %.1fs (%d chars)",
                time.perf_counter() - t0,
                len(full_response),
            )
            self.history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield "Sorry, I could not process that."

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("LLM conversation reset")

    def close(self):
        self._client.close()
