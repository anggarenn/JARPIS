"""Text-to-Speech (TTS) engine.
Primary: Microsoft Edge Neural TTS streaming (zero RAM, natural voice).
Default voices: id-ID-ArdiNeural (Indonesian) / en-US-ChristopherNeural (English).
"""

import asyncio
from typing import AsyncGenerator, Optional
import edge_tts


class TTSEngine:
    """Async Edge-TTS streaming engine with in-memory audio buffers."""

    def __init__(
        self,
        voice: str = "id-ID-ArdiNeural",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume

    async def stream_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream raw MP3 audio chunks as they arrive from edge-tts."""
        clean_text = text.strip()
        if not clean_text:
            return

        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
            volume=self.volume,
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize(self, text: str) -> bytes:
        """Synthesize full sentence into in-memory MP3 bytes."""
        chunks = []
        async for chunk in self.stream_audio(text):
            chunks.append(chunk)
        return b"".join(chunks)

    def synthesize_sync(self, text: str) -> bytes:
        """Synchronous wrapper for synthesize."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self.synthesize(text), loop)
            return future.result()
        else:
            return asyncio.run(self.synthesize(text))
