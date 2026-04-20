"""
LLM Client — Ollama-compatible chat client with streaming support.
Streams responses sentence-by-sentence for ultra-fast TTS pipelining.
"""
import logging
import httpx
import re

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a sweet, cheerful, and caring female voice assistant with a warm, anime-inspired personality. "
    "Speak in a cute, friendly, and slightly playful tone. Keep your responses concise and natural "
    "for spoken conversation — ideally 1-3 sentences. "
    "IMPORTANT: Do NOT use emojis, emoticons, special characters, asterisks, or any formatting. "
    "Your output goes directly to a text-to-speech engine that can ONLY read plain English text. "
    "Use words like 'hee hee' or 'aww' instead of emojis to express emotions. "
    "Be enthusiastic, supportive, and speak naturally with proper punctuation."
)


class LLMClient:
    """Ollama chat client with both blocking and streaming modes."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "gemma3:4b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._client = httpx.Client(timeout=60.0)
        logger.info(f"LLM initialized (url={self.base_url}, model={self.model})")

    def send(self, text: str) -> str:
        """Send user text and return full response (blocking)."""
        try:
            self.history.append({"role": "user", "content": text})
            resp = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": self.history,
                    "max_tokens": 256,
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"].strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I could not process that."

    def stream_sentences(self, text: str):
        """Stream LLM response, yielding complete sentences as they form."""
        try:
            self.history.append({"role": "user", "content": text})
            buffer = ""
            full_response = ""

            with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": self.history,
                    "max_tokens": 256,
                    "temperature": 0.7,
                    "stream": True,
                },
            ) as resp:
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            buffer += token
                            full_response += token

                            # Yield when we hit sentence-ending punctuation
                            sentences = re.split(r'(?<=[.!?])\s+', buffer, maxsplit=1)
                            if len(sentences) > 1:
                                yield sentences[0]
                                buffer = sentences[1]
                    except Exception:
                        continue

            # Yield remaining buffer
            if buffer.strip():
                yield buffer.strip()

            self.history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield "Sorry, I could not process that."

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("LLM conversation reset")

    def close(self):
        self._client.close()
