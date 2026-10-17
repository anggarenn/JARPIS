"""Sanity verification suite for JARPIS local runtime environment.
Non-intrusive tests:
- Test 1: Query audio input/output devices via sounddevice.
- Test 2: In-memory Edge-TTS synthesis validation (no audio playback).
- Test 3: Local 9Router HTTP API connectivity verification.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sounddevice as sd
import httpx
from jarpis.tts import TTSEngine


class TestJarpisSanity(unittest.TestCase):
    """Sanity checks for audio hardware, TTS engine, and 9Router connection."""

    def test_01_audio_devices_query(self):
        """Verify sounddevice detects audio devices on host."""
        devices = sd.query_devices()
        self.assertIsNotNone(devices, "sounddevice.query_devices() returned None")
        self.assertGreater(len(devices), 0, "No audio devices detected")

        input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
        output_devices = [d for d in devices if d.get("max_output_channels", 0) > 0]

        print(f"\n[PASS] Audio Devices: total={len(devices)}, inputs={len(input_devices)}, outputs={len(output_devices)}")
        if input_devices:
            print(f"       Default Input: {input_devices[0]['name']}")
        if output_devices:
            print(f"       Default Output: {output_devices[0]['name']}")

        self.assertGreater(len(input_devices), 0, "No audio input device found")
        self.assertGreater(len(output_devices), 0, "No audio output device found")

    def test_02_edge_tts_in_memory_synthesis(self):
        """Verify Edge-TTS synthesizes audio into in-memory MP3 buffer without disk write or audio playback."""
        engine = TTSEngine(voice="id-ID-ArdiNeural")
        test_phrase = "Sistem JARPIS siap beroperasi."

        mp3_buffer = asyncio.run(engine.synthesize(test_phrase))

        self.assertIsInstance(mp3_buffer, bytes)
        self.assertGreater(len(mp3_buffer), 1024, "Synthesized audio buffer too small (< 1KB)")

        # Verify MP3 sync header (ID3v2 tag 'ID3' or MPEG sync frame 0xFF)
        has_id3 = mp3_buffer.startswith(b"ID3")
        has_mpeg_sync = mp3_buffer[0] == 0xFF and (mp3_buffer[1] & 0xE0) == 0xE0
        self.assertTrue(has_id3 or has_mpeg_sync, f"Buffer missing MP3 header: {mp3_buffer[:10]}")

        print(f"\n[PASS] Edge-TTS In-Memory Synthesis: size={len(mp3_buffer)} bytes, voice={engine.voice}")

    def test_03_9router_connection(self):
        """Verify HTTP connectivity to local 9Router API endpoint."""
        router_url = "http://127.0.0.1:20128/v1/models"
        try:
            response = httpx.get(router_url, timeout=5.0)
        except httpx.ConnectError as err:
            self.fail(f"Failed to connect to 9Router on {router_url}: {err}")

        self.assertEqual(response.status_code, 200, f"Unexpected status code: {response.status_code}")
        data = response.json()
        models = data.get("data", [])

        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0, "No models returned from 9Router")

        model_ids = [m.get("id") for m in models]
        print(f"\n[PASS] 9Router Connectivity: HTTP 200, available_models={model_ids[:5]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
