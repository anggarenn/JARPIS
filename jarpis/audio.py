"""Audio capture and non-blocking playback engine.
Zero-disk retention: All audio buffers remain exclusively in-memory.
"""

import io
import queue
import threading
import wave
from typing import Optional, Tuple
import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    """In-memory 16kHz mono audio recorder using sounddevice.RawInputStream."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream: Optional[sd.RawInputStream] = None
        self._frames: bytearray = bytearray()
        self._is_recording = False
        self._lock = threading.Lock()

    def _audio_callback(self, indata, frame_count, time_info, status):
        if status:
            pass
        if self._is_recording:
            self._frames.extend(indata)

    def start(self):
        """Start capturing audio into memory."""
        with self._lock:
            if self._is_recording:
                return
            self._frames.clear()
            self._is_recording = True
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                callback=self._audio_callback,
            )
            self._stream.start()

    def stop(self) -> bytes:
        """Stop capturing and return raw WAV bytes in memory."""
        with self._lock:
            if not self._is_recording:
                return b""
            self._is_recording = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None

            raw_data = bytes(self._frames)
            self._frames.clear()

        if not raw_data:
            return b""

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(raw_data)

        return buffer.getvalue()


class AudioPlayer:
    """Non-blocking audio player supporting in-memory MP3 and WAV playback."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker_thread.start()

    def _playback_worker(self):
        while not self._stop_event.is_set():
            try:
                audio_bytes = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if audio_bytes is None:
                break

            try:
                with io.BytesIO(audio_bytes) as bio:
                    data, samplerate = sf.read(bio, dtype="float32")
                sd.play(data, samplerate)
                sd.wait()
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def play(self, audio_bytes: bytes):
        """Enqueue audio bytes for non-blocking playback."""
        if audio_bytes:
            self._queue.put(audio_bytes)

    def wait_until_done(self):
        """Wait until all queued audio has finished playing."""
        self._queue.join()

    def stop(self):
        """Stop current playback and clear pending queue."""
        sd.stop()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

    def close(self):
        """Shutdown player thread."""
        self.stop()
        self._stop_event.set()
        self._queue.put(None)


def play_beep(freq: float = 880.0, duration: float = 0.06, volume: float = 0.15):
    """Play a short synthetic beep cue using numpy sine wave."""
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), False)
    sine = np.sin(2 * np.pi * freq * t)
    # Simple linear envelope to eliminate audio clicks
    env_len = int(sr * 0.01)
    env = np.ones_like(sine)
    env[:env_len] = np.linspace(0, 1, env_len)
    env[-env_len:] = np.linspace(1, 0, env_len)
    audio = (sine * env * volume).astype(np.float32)
    sd.play(audio, sr)
