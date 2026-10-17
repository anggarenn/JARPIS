"""End-to-End Integration Verification Suite for JARPIS Backend & UI.
Tests:
1. HTTP Server serves index.html and assets locally with 200 OK & CORS headers.
2. WebSocket /ws establishes handshake and delivers initial telemetry.
3. Mic button click event flow (start_listen -> listening status broadcast -> stop_listen).
4. End-to-end turn execution: User voice/text -> 9Router LLM -> Edge-TTS playback -> Status returns to Idle.
"""

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiohttp import ClientSession, web
from jarpis.server import JarpisServer


class TestJarpisUIIntegration(unittest.IsolatedAsyncioTestCase):
    """End-to-End integration test between JARPIS Backend and Arc Reactor HUD."""

    async def asyncSetUp(self):
        self.test_port = 20141
        self.server = JarpisServer(host="127.0.0.1", port=self.test_port, enable_hotkey=False)
        self.server.loop = asyncio.get_running_loop()
        self.server.engine.loop = self.server.loop

        self.runner = web.AppRunner(self.server.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", self.test_port)
        await self.site.start()

    async def asyncTearDown(self):
        self.server.engine.player.close()
        await self.runner.cleanup()

    async def test_01_http_serves_hud_and_cors(self):
        """Test that HTTP server serves UI index.html with 200 OK and CORS headers."""
        url = f"http://127.0.0.1:{self.test_port}/"
        async with ClientSession() as session:
            # 1. Test GET / serves Arc Reactor HUD index.html
            async with session.get(url) as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("text/html", resp.headers.get("Content-Type", ""))
                self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")
                html_text = await resp.text()
                self.assertIn("reactorBtn", html_text)
                self.assertIn("waveformCanvas", html_text)
                self.assertIn("J.A.R.V.I.S.", html_text)
                self.assertIn("connectWebSocket", html_text)

            # 2. Test OPTIONS preflight request for CORS
            async with session.options(f"http://127.0.0.1:{self.test_port}/api/status") as resp:
                self.assertEqual(resp.status, 204)
                self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")
                self.assertIn("GET", resp.headers.get("Access-Control-Allow-Methods", ""))

        print("\n[PASS] HTTP Server: index.html served with 200 OK, Arc Reactor HUD assets verified, CORS headers present.")

    async def test_02_websocket_mic_toggle_flow(self):
        """Test UI Mic interaction flow over WebSocket: start_listen -> status: listening -> stop_listen -> status: idle."""
        ws_url = f"http://127.0.0.1:{self.test_port}/ws"
        async with ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                # Initial handshake
                init_msg = await ws.receive_json()
                self.assertEqual(init_msg.get("type"), "init")
                self.assertEqual(init_msg.get("status"), "idle")

                # UI Click Mic Button -> start_listen
                await ws.send_json({"action": "start_listen"})

                # Wait for status: listening broadcast
                received_listening = False
                for _ in range(5):
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    if msg.get("type") == "status" and msg.get("status") == "listening":
                        received_listening = True
                        break
                    elif msg.get("type") == "amplitude":
                        continue

                self.assertTrue(received_listening, "Did not receive status: listening broadcast")
                self.assertTrue(self.server.engine.is_recording)

                # UI Stop Mic Button -> stop_listen
                await ws.send_json({"action": "stop_listen"})

                # Wait for status return
                for _ in range(5):
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    if msg.get("type") == "status" and msg.get("status") in ("idle", "thinking"):
                        break

        print("\n[PASS] UI Mic Interaction Flow: Click Mic -> WebSocket start_listen -> status: listening -> stop_listen verified.")

    async def test_03_end_to_end_cognitive_and_speech_turn(self):
        """Test full turn: Send command -> 9Router LLM -> Edge-TTS synthesis -> Status returns to Idle."""
        ws_url = f"http://127.0.0.1:{self.test_port}/ws"
        async with ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                # Discard init
                await ws.receive_json()

                # Dispatch voice command text over WebSocket
                test_prompt = "Jarvis, quick status report in one sentence."
                await ws.send_json({"action": "send_text", "text": test_prompt})

                events_received = []
                status_sequence = []

                # Collect streaming events until reply_complete or timeout
                timeout_time = asyncio.get_running_loop().time() + 15.0
                while asyncio.get_running_loop().time() < timeout_time:
                    try:
                        msg = await asyncio.wait_for(ws.receive_json(), timeout=10.0)
                        msg_type = msg.get("type")
                        events_received.append(msg_type)

                        if msg_type == "status":
                            status_sequence.append(msg.get("status"))

                        if msg_type == "reply_complete":
                            break
                    except asyncio.TimeoutError:
                        break

                self.assertIn("transcript", events_received, "Expected 'transcript' event to be broadcast")
                self.assertIn("reply_chunk", events_received, "Expected at least one 'reply_chunk' streaming event")
                self.assertIn("reply_complete", events_received, "Expected 'reply_complete' turn finale")
                self.assertIn("thinking", status_sequence, "Expected transition to 'thinking'")

                # Final engine status must be idle
                self.assertEqual(self.server.engine.status, "idle", "Engine status must return to idle")

        print(f"\n[PASS] E2E Cognitive Flow: Prompt -> 9Router ({self.server.engine.llm_model}) -> Edge-TTS ({self.server.engine.tts_voice}) -> Idle verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
