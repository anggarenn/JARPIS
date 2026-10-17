"""Text-to-Speech (TTS) engine.
Primary: Microsoft Edge Neural TTS streaming (zero RAM, natural voice).
Default voices: id-ID-ArdiNeural (Indonesian) / en-US-ChristopherNeural (English).
"""

import asyncio
import re
from typing import AsyncGenerator, Optional, Tuple
import edge_tts

# Heuristic markers for rapid language detection without heavy external libraries
INDONESIAN_MARKERS = {
    "siap", "aman", "lagi", "ini", "itu", "sudah", "udah", "ada", "tidak",
    "gak", "nggak", "ngga", "tak", "bisa", "akan", "dengan", "untuk", "dari",
    "ke", "di", "yang", "dan", "saya", "kamu", "kita", "kami", "mereka",
    "dia", "nya", "sedang", "harus", "baik", "terima", "kasih", "sama",
    "jangan", "nanti", "sekarang", "bila", "jika", "kalau", "tapi", "tetapi",
    "karena", "kenapa", "mengapa", "bagaimana", "gimana", "apa", "apakah",
    "boleh", "juga", "perlu", "mau", "ingin", "jadi", "hanya", "cuma",
    "bukan", "halo", "hallo", "hai", "oke", "semua", "banyak", "sedikit",
    "saja", "aja", "dong", "kan", "deh", "kok", "sih", "lah", "nih", "tuh",
    "banget", "cek", "tolong", "bantu", "pagi", "siang", "sore", "malam",
    "beres", "proses", "jalan", "lapor", "tuan", "masih", "tentu", "begitu",
    "bagus", "cepat", "mohon", "silakan", "silahkan", "pak", "mas", "bung"
}

ENGLISH_MARKERS = {
    "the", "is", "are", "you", "and", "in", "to", "of", "with", "all",
    "ready", "checking", "sir", "for", "it", "that", "this", "have",
    "has", "will", "would", "could", "should", "your", "my", "our",
    "we", "they", "he", "she", "at", "be", "been", "being", "do",
    "does", "did", "done", "can", "may", "might", "must", "shall",
    "not", "no", "yes", "on", "by", "as", "from", "about", "into",
    "through", "during", "before", "after", "above", "below", "up",
    "down", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "what",
    "which", "who", "whom", "whose", "pleasure", "service", "system",
    "systems", "online", "running", "offline", "standby"
}


class TTSEngine:
    """Async Edge-TTS streaming engine with dynamic bilingual voice switching."""

    def __init__(
        self,
        voice: str = "en-GB-RyanNeural",
        rate: str = "+3%",
        pitch: str = "-5Hz",
        volume: str = "+0%",
        id_voice: str = "id-ID-ArdiNeural",
        id_rate: str = "+6%",
        id_pitch: str = "-2Hz",
        auto_detect: bool = True,
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume

        # Indonesian voice profile
        self.id_voice = id_voice
        self.id_rate = id_rate
        self.id_pitch = id_pitch
        self.auto_detect = auto_detect

    @staticmethod
    def detect_language(text: str) -> str:
        """Fast heuristic language detection between Indonesian ('id') and English ('en')."""
        if not text or not text.strip():
            return "en"

        tokens = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        if not tokens:
            return "en"

        id_score = 0
        en_score = 0

        for word in tokens:
            if word in INDONESIAN_MARKERS:
                id_score += 2
            elif len(word) > 4:
                if word.endswith(("nya", "kan", "lah", "kah", "pun")):
                    id_score += 1
                elif word.startswith(("ber", "ter", "meng", "mem", "men")):
                    id_score += 1

            if word in ENGLISH_MARKERS:
                en_score += 2

        if id_score > 0 and id_score >= en_score:
            return "id"
        if id_score > en_score:
            return "id"
        return "en"

    def get_voice_params(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Resolve voice, rate, and pitch considering auto language detection and overrides."""
        if voice:
            return voice, rate or self.rate, pitch or self.pitch

        if not self.auto_detect:
            return self.voice, rate or self.rate, pitch or self.pitch

        lang = self.detect_language(text)
        if lang == "id":
            return self.id_voice, rate or self.id_rate, pitch or self.id_pitch
        else:
            return self.voice, rate or self.rate, pitch or self.pitch

    async def stream_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Stream raw MP3 audio chunks with dynamic bilingual voice profile."""
        clean_text = text.strip()
        if not clean_text:
            return

        target_voice, target_rate, target_pitch = self.get_voice_params(
            clean_text, voice=voice, rate=rate, pitch=pitch
        )

        communicate = edge_tts.Communicate(
            text=clean_text,
            voice=target_voice,
            rate=target_rate,
            pitch=target_pitch,
            volume=self.volume,
        )

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ) -> bytes:
        """Synthesize full sentence into in-memory MP3 bytes."""
        chunks = []
        async for chunk in self.stream_audio(text, voice=voice, rate=rate, pitch=pitch):
            chunks.append(chunk)
        return b"".join(chunks)

    def synthesize_sync(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ) -> bytes:
        """Synchronous wrapper for synthesize."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self.synthesize(text, voice=voice, rate=rate, pitch=pitch), loop
            )
            return future.result()
        else:
            return asyncio.run(self.synthesize(text, voice=voice, rate=rate, pitch=pitch))
