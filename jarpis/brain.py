"""OpenAI-compatible LLM streaming client and sentence boundary chunker.
Connects to 9Router (http://127.0.0.1:20128/v1) or any OpenAI-compatible gateway.
"""

import json
import re
from typing import AsyncGenerator, Generator, List, Dict, Optional
import httpx


class SentenceChunker:
    """Chunks incoming token stream into natural sentence units for TTS."""

    SENTENCE_END_REGEX = re.compile(r'([.!?\n]+(?:\s+|$))')

    def __init__(self, min_words_for_comma: int = 8):
        self.min_words_for_comma = min_words_for_comma
        self._buffer = ""

    def feed(self, token: str) -> List[str]:
        """Feed a token delta and return any completed sentences."""
        self._buffer += token
        sentences = []

        while True:
            match = self.SENTENCE_END_REGEX.search(self._buffer)
            if match:
                end_pos = match.end()
                sentence = self._buffer[:end_pos].strip()
                self._buffer = self._buffer[end_pos:]
                if sentence:
                    sentences.append(sentence)
                continue

            # Fallback chunking: comma after sufficient words
            if ", " in self._buffer:
                comma_idx = self._buffer.find(", ")
                segment = self._buffer[:comma_idx].strip()
                words = segment.split()
                if len(words) >= self.min_words_for_comma:
                    sentences.append(segment)
                    self._buffer = self._buffer[comma_idx + 2:]
                    continue

            break

        return sentences

    def finalize(self) -> List[str]:
        """Flush remaining buffer as the final sentence."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return [remaining] if remaining else []


class BrainClient:
    """Async client for 9Router / OpenAI-compatible chat completions."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:20128/v1",
        api_key: str = "dummy",
        model: str = "anthropic/claude-3-5-sonnet-20241022",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def stream_tokens(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream token deltas from chat completion endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

    async def stream_sentences(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream complete sentence chunks as tokens arrive from the model."""
        chunker = SentenceChunker()
        async for token in self.stream_tokens(messages, temperature=temperature):
            for sentence in chunker.feed(token):
                yield sentence

        for sentence in chunker.finalize():
            yield sentence
