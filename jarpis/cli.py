"""Push-to-Talk (PTT) CLI Orchestrator.
F8 Key -> In-Memory Recording -> Groq STT -> 9Router LLM -> Edge-TTS -> Non-blocking Audio Playback.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pynput import keyboard

from jarpis.audio import AudioRecorder, AudioPlayer, play_beep
from jarpis.stt import STTEngine
from jarpis.tts import TTSEngine
from jarpis.brain import BrainClient


def load_env_file(path: str = ".env"):
    """Lightweight .env loader avoiding extra dependencies."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k not in os.environ:
                os.environ[k] = v


class JarpisAssistant:
    """Core Jarvis Voice Assistant orchestrator."""

    def __init__(self):
        load_env_file()

        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo")

        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:20128/v1")
        self.llm_api_key = os.getenv("LLM_API_KEY", "dummy")
        self.llm_model = os.getenv("LLM_MODEL", "anthropic/claude-3-5-sonnet-20241022")
        self.system_prompt = os.getenv(
            "SYSTEM_PROMPT",
            "Kamu adalah Jarvis, asisten AI pribadi. Jawab singkat, padat, teknis, dan natural dalam bahasa Indonesia.",
        )

        self.tts_voice = os.getenv("TTS_VOICE", "id-ID-ArdiNeural")
        self.tts_rate = os.getenv("TTS_RATE", "+0%")
        self.tts_pitch = os.getenv("TTS_PITCH", "+0Hz")

        self.sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        self.enable_cues = os.getenv("ENABLE_AUDIO_CUES", "true").lower() == "true"
        self.target_key = os.getenv("PTT_KEY", "f8").lower()

        # Engines
        self.recorder = AudioRecorder(sample_rate=self.sample_rate)
        self.player = AudioPlayer()
        self.stt = STTEngine(
            groq_api_key=self.groq_api_key,
            groq_model=self.groq_model,
        )
        self.tts = TTSEngine(
            voice=self.tts_voice,
            rate=self.tts_rate,
            pitch=self.tts_pitch,
        )
        self.brain = BrainClient(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
            model=self.llm_model,
        )

        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        self.is_recording = False
        self.is_processing = False
        self.loop: asyncio.AbstractEventLoop = None
        self.event_queue: asyncio.Queue = None

    def _matches_key(self, key) -> bool:
        try:
            if hasattr(key, "name") and key.name and key.name.lower() == self.target_key:
                return True
            if hasattr(key, "char") and key.char and key.char.lower() == self.target_key:
                return True
        except Exception:
            pass
        return False

    def _on_press(self, key):
        if self._matches_key(key):
            if not self.is_recording and not self.is_processing:
                if self.loop and self.event_queue:
                    self.loop.call_soon_threadsafe(self.event_queue.put_nowait, "PRESS")
        elif key == keyboard.Key.enter:
            if not self.is_processing:
                if not self.is_recording:
                    if self.loop and self.event_queue:
                        self.loop.call_soon_threadsafe(self.event_queue.put_nowait, "PRESS")
                else:
                    if self.loop and self.event_queue:
                        self.loop.call_soon_threadsafe(self.event_queue.put_nowait, "RELEASE")

    def _on_release(self, key):
        if self._matches_key(key):
            if self.is_recording:
                if self.loop and self.event_queue:
                    self.loop.call_soon_threadsafe(self.event_queue.put_nowait, "RELEASE")
        elif key == keyboard.Key.esc:
            if self.loop and self.event_queue:
                self.loop.call_soon_threadsafe(self.event_queue.put_nowait, "EXIT")

    async def _handle_recording(self):
        print(f"\n[*] Ready! Hold [{self.target_key.upper()}] to Speak (or tap [ENTER] to toggle) | ESC to Exit")
        while True:
            action = await self.event_queue.get()
            if action == "EXIT":
                print("\n[Exiting JARPIS. Sampai jumpa!]")
                break

            if action == "PRESS" and not self.is_recording:
                # Stop any ongoing playback when user starts talking
                self.player.stop()
                self.is_recording = True
                if self.enable_cues:
                    play_beep(880.0, 0.05)
                self.recorder.start()
                print("\r[*] Listening... (Release [F8] or tap [ENTER] to send) ", end="", flush=True)

            elif action == "RELEASE" and self.is_recording:
                self.is_recording = False
                wav_bytes = self.recorder.stop()
                if self.enable_cues:
                    play_beep(660.0, 0.05)

                print("\r[>] Transcribing speech...         ", end="", flush=True)
                self.is_processing = True
                try:
                    await self._process_turn(wav_bytes)
                finally:
                    self.is_processing = False
                    print(f"\n[*] Hold [{self.target_key.upper()}] to Speak... Ready")

    async def _process_turn(self, wav_bytes: bytes):
        if not wav_bytes:
            print("\r[No audio captured]            ")
            return

        # 1. Speech to Text
        try:
            user_text = await asyncio.to_thread(self.stt.transcribe, wav_bytes)
        except Exception as e:
            print(f"\r[STT Error: {e}]")
            return

        if not user_text:
            print("\r[Silence / No speech recognized]")
            return

        print(f"\r\nYou: {user_text}")
        self.history.append({"role": "user", "content": user_text})

        # 2. Brain Streaming + Sentence Chunker + TTS Streaming
        print("Jarvis: ", end="", flush=True)
        full_reply_parts = []

        try:
            async for sentence in self.brain.stream_sentences(self.history):
                print(sentence, end=" ", flush=True)
                full_reply_parts.append(sentence)

                # Generate TTS audio for sentence chunk
                try:
                    audio_chunk = await self.tts.synthesize(sentence)
                    if audio_chunk:
                        self.player.play(audio_chunk)
                except Exception as tts_err:
                    print(f"\n[TTS Error: {tts_err}]", end="")

            print()
            full_reply = " ".join(full_reply_parts).strip()
            if full_reply:
                self.history.append({"role": "assistant", "content": full_reply})

        except Exception as brain_err:
            print(f"\n[Brain Error: {brain_err}]")

    def run(self):
        """Start the PTT interaction loop."""
        print("=" * 60)
        print(" JARPIS Voice Assistant (Sub-800ms Latency, Zero-CUDA)")
        print(f" - Gateway: {self.llm_base_url} ({self.llm_model})")
        print(f" - Voice:   {self.tts_voice}")
        print(f" - Trigger: Hold [{self.target_key.upper()}] or tap [ENTER] | ESC to Exit")
        print("=" * 60)

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.event_queue = asyncio.Queue()

        listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        listener.start()

        try:
            self.loop.run_until_complete(self._handle_recording())
        except KeyboardInterrupt:
            print("\n[Terminated by user]")
        finally:
            listener.stop()
            self.player.close()
            self.loop.close()


def main():
    assistant = JarpisAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
