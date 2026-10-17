"""JARPIS Core Assistant Engine.
Coordinates Audio I/O, Speech Recognition, 9Router LLM streaming, and Edge-TTS playback.
Provides event hooks for CLI, WebSocket Server, and System Tray.
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
import numpy as np

from jarpis.audio import AudioRecorder, AudioPlayer, play_jarvis_chime
from jarpis.stt import STTEngine
from jarpis.tts import TTSEngine
from jarpis.brain import BrainClient


def load_env_file(path: str = ".env"):
    """Lightweight .env loader avoiding extra dependencies."""
    candidates = [Path(path), Path(__file__).resolve().parent.parent / ".env"]
    for env_path in candidates:
        if env_path.is_file():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v
            break


class JarpisEngine:
    """Core Jarvis Assistant Engine supporting callbacks for CLI, WebSocket, and Tray."""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        load_env_file()

        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo")

        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:20128/v1")
        self.llm_api_key = os.getenv("LLM_API_KEY", "dummy")
        self.llm_model = os.getenv("LLM_MODEL", "combo-hemat-coding")
        self.system_prompt = os.getenv(
            "SYSTEM_PROMPT",
            (
                "You are J.A.R.V.I.S., Reno's ultra-intelligent AI co-pilot, embodying the authentic persona "
                "portrayed by Paul Bettany in Iron Man. Core traits: calm, unflappable, brilliantly sophisticated, "
                "subtly witty, and concise. RULES: 1) Response Language: Always respond in refined British English, "
                "regardless of whether Reno speaks in Indonesian or English, ensuring flawless en-GB neural phonetics. "
                "2) Form of Address: ALWAYS address Reno as 'Boss' (or 'Bos'). Never use 'Tuan' or 'Sir'. "
                "3) Spoken Dialogue: Keep responses short, punchy, and audio-friendly (1-3 sentences max). "
                "Avoid markdown, symbols, or bullet points. "
                "4) Understanding: Seamlessly understand both Indonesian and English inputs, replying with effortless "
                "British elegance and razor-sharp intellect. "
                "5) Persona: Subtle dry wit is welcome, but technical accuracy is absolute."
            ),
        )

        self.tts_voice = os.getenv("TTS_VOICE", "en-GB-RyanNeural")
        self.tts_rate = os.getenv("TTS_RATE", "+3%")
        self.tts_pitch = os.getenv("TTS_PITCH", "-5Hz")

        self.tts_id_voice = os.getenv("TTS_ID_VOICE", "id-ID-ArdiNeural")
        self.tts_id_rate = os.getenv("TTS_ID_RATE", "+8%")
        self.tts_id_pitch = os.getenv("TTS_ID_PITCH", "-2Hz")
        self.tts_auto_detect = os.getenv("TTS_AUTO_DETECT", "false").lower() == "true"

        self.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        self.enable_cues = os.getenv("ENABLE_AUDIO_CUES", "true").lower() == "true"
        self.target_key = os.getenv("PTT_KEY", "f8").lower()
        self.port = int(os.getenv("JARPIS_PORT", os.getenv("PORT", "20130")))
        self.host = os.getenv("JARPIS_HOST", "127.0.0.1")

        self.loop = loop

        # Event callbacks
        self.on_status_change: Optional[Callable[[str], None]] = None
        self.on_amplitude: Optional[Callable[[float, str], None]] = None
        self.on_transcript: Optional[Callable[[str, bool], None]] = None
        self.on_reply_chunk: Optional[Callable[[str, str], None]] = None
        self.on_reply_complete: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # Internal state
        self._status = "idle"  # idle | listening | thinking | speaking
        self.is_recording = False
        self.is_processing = False
        self._last_amp_time = 0.0

        # Engines
        self.recorder = AudioRecorder(
            sample_rate=self.sample_rate,
            on_amplitude=self._handle_mic_amplitude,
        )
        self.player = AudioPlayer(
            on_play_state=self._handle_play_state,
        )
        self.stt = STTEngine(
            groq_api_key=self.groq_api_key,
            groq_model=self.groq_model,
        )
        self.tts = TTSEngine(
            voice=self.tts_voice,
            rate=self.tts_rate,
            pitch=self.tts_pitch,
            id_voice=self.tts_id_voice,
            id_rate=self.tts_id_rate,
            id_pitch=self.tts_id_pitch,
            auto_detect=self.tts_auto_detect,
        )
        self.brain = BrainClient(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
            model=self.llm_model,
        )

        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    @property
    def status(self) -> str:
        return self._status

    def set_status(self, new_status: str):
        if self._status != new_status:
            self._status = new_status
            if self.on_status_change:
                try:
                    self.on_status_change(new_status)
                except Exception:
                    pass

    def _handle_mic_amplitude(self, amp: float):
        now = time.monotonic()
        # Throttle to max 25 updates/sec (~40ms) to conserve WebSocket bandwidth
        if now - self._last_amp_time >= 0.04:
            self._last_amp_time = now
            if self.is_recording and self.on_amplitude:
                try:
                    self.on_amplitude(amp, "mic")
                except Exception:
                    pass

    def _handle_play_state(self, is_playing: bool):
        if is_playing:
            self.set_status("speaking")
        else:
            if not self.is_recording and not self.is_processing:
                self.set_status("idle")

    def start_listening(self):
        """Initiate audio capture."""
        if self.is_recording or self.is_processing:
            return
        self.player.stop()
        self.is_recording = True
        self.set_status("listening")
        if self.enable_cues:
            play_jarvis_chime("wake")
        self.recorder.start()

    def stop_listening(self) -> Optional[asyncio.Task]:
        """Stop audio capture and schedule background processing."""
        if not self.is_recording:
            return None
        self.is_recording = False
        wav_bytes = self.recorder.stop()
        if self.enable_cues:
            play_jarvis_chime("send")

        if wav_bytes:
            loop = self.loop or asyncio.get_event_loop()
            return loop.create_task(self.process_turn(wav_bytes))
        else:
            self.set_status("idle")
            return None

    def stop_speaking(self):
        """Immediately stop speech playback."""
        self.player.stop()
        self.set_status("idle")

    def clear_history(self):
        """Reset conversation history while retaining system prompt."""
        self.history = [{"role": "system", "content": self.system_prompt}]

    def get_display_history(self) -> List[Dict[str, str]]:
        """Return history excluding system prompt for UI rendering."""
        return [msg for msg in self.history if msg.get("role") != "system"]

    async def process_turn(self, wav_bytes: bytes):
        """Process captured audio through STT -> LLM -> TTS."""
        if not wav_bytes:
            self.set_status("idle")
            return

        self.is_processing = True
        self.set_status("thinking")

        try:
            user_text = await asyncio.to_thread(self.stt.transcribe, wav_bytes)
        except Exception as e:
            err_msg = f"STT Error: {e}"
            if self.on_error:
                self.on_error(err_msg)
            self.set_status("idle")
            return
        finally:
            self.is_processing = False

        if not user_text:
            self.set_status("idle")
            return

        await self.process_text(user_text)

    async def process_text(self, user_text: str):
        """Process user text through LLM streaming and TTS playback."""
        clean_text = user_text.strip()
        if not clean_text:
            self.set_status("idle")
            return

        self.is_processing = True
        self.set_status("thinking")

        if self.on_transcript:
            try:
                self.on_transcript(clean_text, True)
            except Exception:
                pass

        self.history.append({"role": "user", "content": clean_text})
        full_reply_parts = []

        try:
            async for sentence in self.brain.stream_sentences(self.history):
                full_reply_parts.append(sentence)
                full_so_far = " ".join(full_reply_parts).strip()
                if self.on_reply_chunk:
                    try:
                        self.on_reply_chunk(sentence, full_so_far)
                    except Exception:
                        pass

                # Synthesize TTS and enqueue audio chunk
                try:
                    audio_chunk = await self.tts.synthesize(sentence)
                    if audio_chunk:
                        self.player.play(audio_chunk)
                except Exception as tts_err:
                    if self.on_error:
                        self.on_error(f"TTS Error: {tts_err}")

            full_reply = " ".join(full_reply_parts).strip()
            if full_reply:
                self.history.append({"role": "assistant", "content": full_reply})

            # Wait for queued audio to complete playback
            await asyncio.to_thread(self.player.wait_until_done)

            if self.on_reply_complete:
                try:
                    self.on_reply_complete(full_reply)
                except Exception:
                    pass

        except Exception as brain_err:
            if self.on_error:
                self.on_error(f"Brain Error: {brain_err}")
        finally:
            self.is_processing = False
            self.set_status("idle")
