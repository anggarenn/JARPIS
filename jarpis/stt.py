"""Speech-to-Text (STT) module.
Primary: Ultra-fast Groq Whisper API (sub-150ms roundtrip).
Fallback: Local faster-whisper running on CPU INT8 (zero-CUDA).
"""

import io
import logging
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class STTEngine:
    """Hybrid STT engine with cloud speed and offline fallback."""

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        groq_model: str = "whisper-large-v3-turbo",
        groq_timeout: float = 3.0,
        enable_fallback: bool = True,
    ):
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self.groq_timeout = groq_timeout
        self.enable_fallback = enable_fallback
        self._local_model = None

    def _is_valid_groq_key(self) -> bool:
        if not self.groq_api_key:
            return False
        key = self.groq_api_key.strip().lower()
        return not (key.startswith("your_") or key.startswith("dummy") or "placeholder" in key)

    def _transcribe_groq(self, wav_bytes: bytes) -> Optional[str]:
        if not self._is_valid_groq_key():
            return None

        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.groq_api_key}"}
        data = {
            "model": self.groq_model,
            "response_format": "json",
            "temperature": "0.0",
        }
        files = {
            "file": ("speech.wav", wav_bytes, "audio/wav"),
        }

        with httpx.Client(timeout=self.groq_timeout) as client:
            resp = client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
            result = resp.json()
            return result.get("text", "").strip()

    def _transcribe_local(self, wav_bytes: bytes) -> str:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Groq STT failed/unavailable, and 'faster-whisper' is not installed for offline fallback."
            ) from exc

        if self._local_model is None:
            logger.info("Initializing faster-whisper CPU INT8 model (4 threads)...")
            self._local_model = WhisperModel(
                "base",
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )

        import soundfile as sf
        audio_data, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        segments, _ = self._local_model.transcribe(audio_data, beam_size=1, language=None)
        text = " ".join(seg.text for seg in segments).strip()
        return text

    def transcribe(self, wav_bytes: bytes) -> str:
        """Transcribe in-memory WAV bytes to text."""
        if not wav_bytes:
            return ""

        # 1. Attempt Groq Whisper Cloud (fastest)
        if self._is_valid_groq_key():
            try:
                text = self._transcribe_groq(wav_bytes)
                if text:
                    return text
            except Exception as err:
                logger.warning(f"Groq STT failed: {err}. Attempting fallback...")

        # 2. Offline Fallback
        if self.enable_fallback:
            return self._transcribe_local(wav_bytes)

        raise RuntimeError("STT transcription failed and fallback is disabled.")
