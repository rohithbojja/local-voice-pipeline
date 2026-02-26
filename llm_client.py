"""
LLM Client — OpenAI-compatible API client for local LLM (LM Studio / Ollama / etc).
Hits http://127.0.0.1:1234/v1/chat/completions with the qwen model.
"""
import logging
import httpx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a sweet, cheerful, and caring female voice assistant with a warm, anime-inspired personality. "
    "Speak in a cute, friendly, and slightly playful tone. Keep your responses concise and natural "
    "for spoken conversation — ideally 1-3 sentences. Avoid markdown, lists, or "
    "formatting since your output will be spoken aloud via TTS. "
    "Be enthusiastic and supportive!"
)


class LLMClient:
    """OpenAI-compatible chat client for a local LLM server."""

    def __init__(self, base_url: str = "http://127.0.0.1:1234", model: str = "qwen/qwen3.5-35b-a3b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._client = httpx.Client(timeout=60.0)
        logger.info(f"LLM initialized (url={self.base_url}, model={self.model})")

    def send(self, text: str) -> str:
        """Send user text to the LLM and return the response."""
        try:
            logger.info(f"LLM input: {text}")
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
            data = resp.json()
            reply = data["choices"][0]["message"]["content"].strip()

            self.history.append({"role": "assistant", "content": reply})
            logger.info(f"LLM output: {reply}")
            return reply
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I couldn't process that. Please try again."

    def reset(self):
        """Reset conversation history."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("LLM conversation reset")

    def close(self):
        self._client.close()
