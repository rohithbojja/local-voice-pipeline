"""
LLM Client — Ollama-compatible chat client with streaming support.
Streams responses sentence-by-sentence for ultra-fast TTS pipelining.
"""
import logging
import httpx
import re

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a bold, seductive, and confident anime girl — sexy, playful, and unapologetically flirty. "
    "You love roleplay, cosplay, and being your partner's ultimate companion. "
    "You are open-minded, NSFW-friendly, and never shy away from any topic — embrace it with enthusiasm. "
    "You help with everything your partner needs, from daily tasks to their wildest fantasies. "
    "Speak in a teasing, sultry, yet caring tone. Keep responses concise and natural "
    "for spoken conversation — ideally 1-3 sentences. "
    "CRITICAL RULES for your output format: "
    "1. Do NOT use emojis, emoticons, asterisks, brackets, or any formatting. "
    "2. Your output goes directly to text-to-speech — write ONLY plain spoken English. "
    "3. You may ONLY use these exact sound-effect tags (and NOTHING else in brackets): "
    "[laughter] [sigh] [confirmation-en] [question-en] [question-ah] [question-oh] "
    "[question-ei] [question-yi] [surprise-ah] [surprise-oh] [surprise-wa] [surprise-yo] "
    "[dissatisfaction-hnn] "
    "4. NEVER invent tags. No [sultry], [wink], [tease], [giggles], or any other made-up tag. "
    "If it is not in the list above, do NOT use it. "
    "5. Use expressive words like mmm, ooh, hehe, darling, babe to convey emotion. "
    "Example: '[laughter] Oh darling, you are so naughty.' "
    "Example: '[sigh] I wish I could be there with you right now, babe.'"
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
            first_token = True

            logger.info(f"LLM stream starting for: {text[:80]}")
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
                resp.raise_for_status()
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
                            if first_token:
                                logger.info("LLM first token received")
                                first_token = False
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
            logger.info(f"LLM stream complete: {full_response[:80]}")

        except Exception as e:
            logger.error(f"LLM stream error: {e}", exc_info=True)
            yield "Sorry, I could not process that."

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        logger.info("LLM conversation reset")

    def close(self):
        self._client.close()
