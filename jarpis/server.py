"""JARPIS Async UI & Telemetry WebSocket Server.
Lightweight aiohttp server broadcasting real-time voice assistant telemetry:
- WebSocket /ws: status ('idle' | 'listening' | 'thinking' | 'speaking'), audio amplitude, transcript, streaming tokens.
- REST /api: status query, direct text input, speech synthesis, and playback control.
- Static HUD: Serves futuristic Holographic UI at http://127.0.0.1:20130.
"""

import asyncio
import json
import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import Set, Optional

from aiohttp import web, WSMsgType
from pynput import keyboard

from jarpis.engine import JarpisEngine

logger = logging.getLogger("jarpis.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class JarpisServer:
    """Async web and WebSocket server for JARPIS HUD interface."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, enable_hotkey: bool = True):
        self.engine = JarpisEngine()
        self.host = host or self.engine.host
        self.port = port or self.engine.port
        self.enable_hotkey = enable_hotkey

        self.app = web.Application()
        self.active_websockets: Set[web.WebSocketResponse] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._listener: Optional[keyboard.Listener] = None

        @web.middleware
        async def cors_middleware(request, handler):
            if request.method == "OPTIONS":
                response = web.Response(status=204)
            else:
                response = await handler(request)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return response

        self.app.middlewares.append(cors_middleware)

        self._setup_routes()
        self._wire_engine_callbacks()

    def _setup_routes(self):
        """Register HTTP and WebSocket routes."""
        self.app.router.add_get("/ws", self.handle_websocket)
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/api/status", self.handle_api_status)
        self.app.router.add_get("/api/history", self.handle_api_history)
        self.app.router.add_post("/api/chat", self.handle_api_chat)
        self.app.router.add_post("/api/speak", self.handle_api_speak)
        self.app.router.add_post("/api/stop", self.handle_api_stop)
        self.app.router.add_post("/api/clear", self.handle_api_clear)

        # Static assets (CSS, JS, audio, images)
        for cand in ["ui", "web"]:
            cand_dir = Path(__file__).parent / cand
            if cand_dir.exists():
                self.app.router.add_static("/static/", path=str(cand_dir), name=f"static_{cand}")
                break

    def _wire_engine_callbacks(self):
        """Connect JarpisEngine event hooks to WebSocket broadcast."""
        def safe_broadcast(msg: dict):
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast(msg), self.loop)

        self.engine.on_status_change = lambda status: safe_broadcast({
            "type": "status",
            "status": status,
        })

        self.engine.on_amplitude = lambda val, src: safe_broadcast({
            "type": "amplitude",
            "value": round(val, 3),
            "source": src,
        })

        self.engine.on_transcript = lambda text, is_final: safe_broadcast({
            "type": "transcript",
            "text": text,
            "is_final": is_final,
        })

        self.engine.on_reply_chunk = lambda chunk, full: safe_broadcast({
            "type": "reply_chunk",
            "chunk": chunk,
            "full_text": full,
        })

        self.engine.on_reply_complete = lambda full: safe_broadcast({
            "type": "reply_complete",
            "full_text": full,
        })

        self.engine.on_error = lambda err: safe_broadcast({
            "type": "error",
            "message": err,
        })

    async def broadcast(self, message: dict):
        """Send JSON payload to all active WebSocket clients."""
        if not self.active_websockets:
            return
        payload = json.dumps(message)
        dead = []
        for ws in self.active_websockets:
            if ws.closed:
                dead.append(ws)
                continue
            try:
                await ws.send_str(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.active_websockets.discard(ws)

    async def handle_index(self, request: web.Request) -> web.Response:
        """Serve HUD HTML interface."""
        for cand in ["ui", "web"]:
            index_file = Path(__file__).parent / cand / "index.html"
            if index_file.is_file():
                return web.FileResponse(str(index_file))
        return web.Response(
            text="""<!DOCTYPE html>
<html>
<head><title>JARPIS HUD Initializing</title></head>
<body style="background:#050b14;color:#00e5ff;font-family:sans-serif;text-align:center;padding:50px;">
  <h2>JARPIS HUD Interface Ready</h2>
  <p>Status: WebSocket online at ws://127.0.0.1:20130/ws</p>
  <p>Awaiting HUD Template installation...</p>
</body>
</html>""",
            content_type="text/html",
        )

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle bi-directional HUD WebSocket connection."""
        ws = web.WebSocketResponse(heartbeat=15.0)
        await ws.prepare(request)
        self.active_websockets.add(ws)
        logger.info(f"HUD WebSocket connected: {request.remote}")

        # Send initial handshake state
        init_payload = {
            "type": "init",
            "status": self.engine.status,
            "model": self.engine.llm_model,
            "voice": self.engine.tts_voice,
            "ptt_key": self.engine.target_key,
            "sample_rate": self.engine.sample_rate,
            "mic": self.engine.recorder.device_name,
            "speaker": self.engine.player.device_name,
            "history": self.engine.get_display_history(),
        }
        await ws.send_str(json.dumps(init_payload))

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue

                    action = data.get("action")
                    if action == "start_listen":
                        self.engine.start_listening()
                    elif action == "stop_listen":
                        self.engine.stop_listening()
                    elif action == "send_text":
                        text = data.get("text", "")
                        if text:
                            asyncio.create_task(self.engine.process_text(text))
                    elif action == "stop_speaking":
                        self.engine.stop_speaking()
                    elif action == "clear_history":
                        self.engine.clear_history()
                        await self.broadcast({"type": "history_cleared"})
                    elif action == "get_status":
                        await ws.send_str(json.dumps({
                            "type": "status",
                            "status": self.engine.status,
                            "model": self.engine.llm_model,
                            "voice": self.engine.tts_voice,
                        }))

                elif msg.type == WSMsgType.ERROR:
                    logger.warning(f"WebSocket closed with exception {ws.exception()}")
        finally:
            self.active_websockets.discard(ws)
            logger.info("HUD WebSocket disconnected.")

        return ws

    async def handle_api_status(self, request: web.Request) -> web.Response:
        """GET /api/status - Return full engine and hardware status."""
        return web.json_response({
            "status": self.engine.status,
            "model": self.engine.llm_model,
            "voice": self.engine.tts_voice,
            "mic": self.engine.recorder.device_name,
            "speaker": self.engine.player.device_name,
            "ptt_key": self.engine.target_key,
            "is_recording": self.engine.is_recording,
            "is_processing": self.engine.is_processing,
            "active_clients": len(self.active_websockets),
        })

    async def handle_api_history(self, request: web.Request) -> web.Response:
        """GET /api/history - Return current chat history."""
        return web.json_response({
            "history": self.engine.get_display_history()
        })

    async def handle_api_chat(self, request: web.Request) -> web.Response:
        """POST /api/chat - Send text prompt to JARPIS."""
        try:
            body = await request.json()
            message = body.get("message", "").strip()
            if not message:
                return web.json_response({"error": "Empty message"}, status=400)

            # Fire and forget turn in background
            asyncio.create_task(self.engine.process_text(message))
            return web.json_response({"status": "accepted", "message": message})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_speak(self, request: web.Request) -> web.Response:
        """POST /api/speak - Synthesize and speak text directly."""
        try:
            body = await request.json()
            text = body.get("text", "").strip()
            if not text:
                return web.json_response({"error": "Empty text"}, status=400)

            async def _speak():
                audio = await self.engine.tts.synthesize(text)
                if audio:
                    self.engine.player.play(audio)

            asyncio.create_task(_speak())
            return web.json_response({"status": "synthesizing", "text": text})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_stop(self, request: web.Request) -> web.Response:
        """POST /api/stop - Stop ongoing speech playback."""
        self.engine.stop_speaking()
        return web.json_response({"status": "stopped"})

    async def handle_api_clear(self, request: web.Request) -> web.Response:
        """POST /api/clear - Clear conversation history."""
        self.engine.clear_history()
        await self.broadcast({"type": "history_cleared"})
        return web.json_response({"status": "cleared"})

    def _setup_hotkey(self):
        """Global push-to-talk keyboard listener (F8 / ENTER)."""
        target = self.engine.target_key.lower()

        def matches_key(key):
            try:
                if hasattr(key, "name") and key.name and key.name.lower() == target:
                    return True
                if hasattr(key, "char") and key.char and key.char.lower() == target:
                    return True
            except Exception:
                pass
            return False

        def on_press(key):
            if matches_key(key):
                if not self.engine.is_recording and not self.engine.is_processing:
                    self.engine.start_listening()
            elif key == keyboard.Key.enter:
                if not self.engine.is_processing:
                    if not self.engine.is_recording:
                        self.engine.start_listening()
                    else:
                        self.engine.stop_listening()

        def on_release(key):
            if matches_key(key):
                if self.engine.is_recording:
                    self.engine.stop_listening()

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        logger.info(f"Global hotkey listener active on [{target.upper()}] / [ENTER]")

    async def start(self):
        """Start aiohttp server asynchronously."""
        self.loop = asyncio.get_running_loop()
        self.engine.loop = self.loop

        if self.enable_hotkey:
            self._setup_hotkey()

        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(f"JARPIS Server running at http://{self.host}:{self.port}")
        logger.info(f"WebSocket endpoint at ws://{self.host}:{self.port}/ws")

    def run_forever(self, open_browser: bool = False):
        """Convenience method to run server synchronously."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.start())
            if open_browser:
                webbrowser.open(f"http://{self.host}:{self.port}")
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Server shutting down by user...")
        finally:
            if self._listener:
                self._listener.stop()
            self.engine.player.close()
            loop.close()


def main():
    """Main CLI entrypoint for running server directly."""
    server = JarpisServer()
    server.run_forever(open_browser=True)


if __name__ == "__main__":
    main()
