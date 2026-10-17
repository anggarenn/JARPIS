"""Audio capture and non-blocking playback engine.
Zero-disk retention: All audio buffers remain exclusively in-memory.
"""

import io
import queue
import threading
import wave
from typing import Optional, Tuple, Callable
import os
import numpy as np
import sounddevice as sd
import soundfile as sf


def find_system_audio_device(kind: str = "output") -> Optional[int]:
    """Find the best audio device that tracks Windows defaults (DirectSound / Sound Mapper)."""
    env_name = os.getenv("AUDIO_OUTPUT_DEVICE" if kind == "output" else "AUDIO_INPUT_DEVICE", "").strip()
    devices = sd.query_devices()

    # 1. Custom device configured via env
    if env_name:
        for idx, d in enumerate(devices):
            if env_name.lower() in d["name"].lower():
                if kind == "output" and d["max_output_channels"] > 0:
                    return idx
                if kind == "input" and d["max_input_channels"] > 0:
                    return idx

    # 2. Windows Primary Sound Driver (DirectSound - tracks active Windows default)
    primary_target = "Primary Sound Driver" if kind == "output" else "Primary Sound Capture Driver"
    for idx, d in enumerate(devices):
        if primary_target.lower() in d["name"].lower():
            return idx

    # 3. Microsoft Sound Mapper (MME - tracks active Windows default)
    mapper_target = "Microsoft Sound Mapper"
    for idx, d in enumerate(devices):
        if mapper_target.lower() in d["name"].lower():
            if kind == "output" and d["max_output_channels"] > 0:
                return idx
            if kind == "input" and d["max_input_channels"] > 0:
                return idx

    # 4. Fallback to PortAudio default
    return sd.default.device[1] if kind == "output" else sd.default.device[0]


class AudioRecorder:
    """In-memory 16kHz mono audio recorder using sounddevice.RawInputStream."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[int] = None,
        on_amplitude: Optional[Callable[[float], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device if device is not None else find_system_audio_device("input")
        self.on_amplitude = on_amplitude
        self._stream: Optional[sd.RawInputStream] = None
        self._frames: bytearray = bytearray()
        self._is_recording = False
        self._lock = threading.Lock()

    @property
    def device_name(self) -> str:
        try:
            return sd.query_devices(self.device)["name"]
        except Exception:
            return f"Device #{self.device}"

    def _audio_callback(self, indata, frame_count, time_info, status):
        if status:
            pass
        if self._is_recording:
            self._frames.extend(indata)
            if self.on_amplitude:
                try:
                    arr = np.frombuffer(indata, dtype=np.int16)
                    if len(arr) > 0:
                        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                        amp = min(1.0, max(0.0, rms / 6000.0))
                        self.on_amplitude(amp)
                except Exception:
                    pass

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
                device=self.device,
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

    def __init__(
        self,
        device: Optional[int] = None,
        on_play_state: Optional[Callable[[bool], None]] = None,
    ):
        self.device = device if device is not None else find_system_audio_device("output")
        self.on_play_state = on_play_state
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._is_playing = False
        self._worker_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._worker_thread.start()

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def device_name(self) -> str:
        try:
            return sd.query_devices(self.device)["name"]
        except Exception:
            return f"Device #{self.device}"

    def _playback_worker(self):
        while not self._stop_event.is_set():
            try:
                audio_bytes = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if audio_bytes is None:
                break

            try:
                self._is_playing = True
                if self.on_play_state:
                    self.on_play_state(True)
                with io.BytesIO(audio_bytes) as bio:
                    data, samplerate = sf.read(bio, dtype="float32")
                sd.play(data, samplerate, device=self.device)
                sd.wait()
            except Exception as err:
                print(f"\n[Audio Playback Warning: {err}]", flush=True)
            finally:
                self._is_playing = False
                if self.on_play_state and self._queue.empty():
                    self.on_play_state(False)
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


def play_jarvis_chime(kind: str = "wake"):
    """Play futuristic Stark Industries holographic HUD chime."""
    sr = 24000
    if kind == "wake":
        # Dual-harmonic ascending sci-fi chime (587Hz D5 -> 880Hz A5 -> 1174Hz D6)
        dur = 0.18
        t = np.linspace(0, dur, int(sr * dur), False)
        wave = (
            0.15 * np.sin(2 * np.pi * 587.33 * t)
            + 0.20 * np.sin(2 * np.pi * 880.00 * t)
            + 0.15 * np.sin(2 * np.pi * 1174.66 * t)
        )
        env = np.exp(-t * 12)
        audio = (wave * env).astype(np.float32)
    elif kind == "send":
        # Soft acknowledgment ping (1046Hz -> decay)
        dur = 0.12
        t = np.linspace(0, dur, int(sr * dur), False)
        wave = 0.18 * np.sin(2 * np.pi * 1046.50 * t) + 0.08 * np.sin(2 * np.pi * 2093.00 * t)
        env = np.exp(-t * 18)
        audio = (wave * env).astype(np.float32)
    else:
        return

    dev = find_system_audio_device("output")
    try:
        sd.play(audio, sr, device=dev)
    except Exception:
        sd.play(audio, sr)


def play_beep(freq: float = 880.0, duration: float = 0.06, volume: float = 0.15):
    """Play a short synthetic cue using sci-fi HUD chime."""
    play_jarvis_chime("wake" if freq >= 800.0 else "send")
