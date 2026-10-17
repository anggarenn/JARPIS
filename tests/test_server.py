"""Verification test suite for JARPIS aiohttp Server and WebSocket Protocol."""

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiohttp import ClientSession, WSMsgType
from jarpis.server import JarpisServer


class TestJarpisServer(unittest.IsolatedAsyncioTestCase):
    """Test server initialization, REST APIs, and WebSocket protocol."""

    async def asyncSetUp(self):
        # Bind to localhost on test port 20139 without activating keyboard hotkey in headless test
        self.test_port = 20139
        self.server = JarpisServer(host="127.0.0.1", port=self.test_port, enable_hotkey=False)
        self.server.loop = asyncio.get_running_loop()
        self.server.engine.loop = self.server.loop

        from aiohttp import web
        self.runner = web.AppRunner(self.server.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", self.test_port)
        await self.site.start()

    async def asyncTearDown(self):
        self.server.engine.player.close()
        await self.runner.cleanup()

    async def test_01_api_status(self):
        """Verify REST /api/status endpoint returns valid JSON metadata."""
        url = f"http://127.0.0.1:{self.test_port}/api/status"
        async with ClientSession() as session:
            async with session.get(url) as resp:
                self.assertEqual(resp.status, 200)
                data = await resp.json()
                self.assertIn("status", data)
                self.assertEqual(data["status"], "idle")
                self.assertIn("model", data)
                self.assertIn("voice", data)
                self.assertIn("mic", data)
                self.assertIn("speaker", data)
                print(f"\n[PASS] /api/status: status={data['status']}, model={data['model']}")

    async def test_02_websocket_handshake_and_protocol(self):
        """Verify WebSocket /ws handshake, initial telemetry payload, and command action."""
        ws_url = f"http://127.0.0.1:{self.test_port}/ws"
        async with ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                # 1. First message must be 'init' handshake
                msg = await ws.receive_json()
                self.assertEqual(msg.get("type"), "init")
                self.assertEqual(msg.get("status"), "idle")
                self.assertIn("model", msg)
                self.assertIn("voice", msg)
                self.assertIn("history", msg)

                # 2. Send ping action 'get_status'
                await ws.send_json({"action": "get_status"})
                status_msg = await ws.receive_json()
                self.assertEqual(status_msg.get("type"), "status")
                self.assertEqual(status_msg.get("status"), "idle")

                # 3. Test clear history action
                await ws.send_json({"action": "clear_history"})
                clear_msg = await ws.receive_json()
                self.assertEqual(clear_msg.get("type"), "history_cleared")

                print(f"\n[PASS] /ws Handshake & Protocol: verified init and action dispatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
