"""JARPIS Windows System Tray Daemon & Background Runner.
Runs JARPIS in the Windows Notification Area with a Stark Industries Arc Reactor icon.
Provides single-click access to the Holographic HUD, status indicator, and voice controls.
"""

import asyncio
import os
import sys
import threading
import webbrowser
from typing import Optional

from PIL import Image, ImageDraw
import pystray
from pystray import MenuItem as item, Menu

from jarpis.server import JarpisServer


def create_arc_reactor_icon(size: int = 64, color: str = "#00e5ff") -> Image.Image:
    """Generate high-DPI Stark Arc Reactor icon for Windows System Tray."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    center = size / 2.0
    r_outer = size * 0.45
    r_mid = size * 0.32
    r_inner = size * 0.16

    # Outer dark chassis ring
    draw.ellipse(
        [center - r_outer, center - r_outer, center + r_outer, center + r_outer],
        outline="#04121f",
        width=int(size * 0.08),
    )

    # Primary neon glow ring
    draw.ellipse(
        [center - r_mid, center - r_mid, center + r_mid, center + r_mid],
        outline=color,
        width=int(size * 0.06),
    )

    # Inner core glow
    draw.ellipse(
        [center - r_inner, center - r_inner, center + r_inner, center + r_inner],
        fill=color,
        outline="#ffffff",
        width=1,
    )

    # 4 Cardinal reactor segment ticks
    tick_len = size * 0.10
    draw.line([center, center - r_outer, center, center - r_outer + tick_len], fill=color, width=2)
    draw.line([center, center + r_outer, center, center + r_outer - tick_len], fill=color, width=2)
    draw.line([center - r_outer, center, center - r_outer + tick_len, center], fill=color, width=2)
    draw.line([center + r_outer, center, center + r_outer - tick_len, center], fill=color, width=2)

    return img


class JarpisTrayApp:
    """System Tray controller for JARPIS background execution."""

    def __init__(self):
        self.server = JarpisServer(enable_hotkey=True)
        self.icon: Optional[pystray.Icon] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()

        # Wire engine status change to tray tooltip and icon update
        self._orig_on_status = self.server.engine.on_status_change

        def on_status_wrapper(status: str):
            if self._orig_on_status:
                self._orig_on_status(status)
            self._update_tray_state(status)

        self.server.engine.on_status_change = on_status_wrapper

    def _update_tray_state(self, status: str):
        if not self.icon:
            return
        colors = {
            "idle": "#00e5ff",       # Cyan
            "listening": "#ffaa00",  # Amber / Gold
            "thinking": "#a855f7",   # Holographic Violet
            "speaking": "#00f59b",   # Energy Green
        }
        color = colors.get(status, "#00e5ff")
        try:
            self.icon.icon = create_arc_reactor_icon(64, color)
            self.icon.title = f"JARPIS: {status.capitalize()} (F8 to speak)"
        except Exception:
            pass

    def open_hud(self):
        """Open web HUD in user's default browser."""
        url = f"http://{self.server.host}:{self.server.port}"
        webbrowser.open(url)

    def trigger_voice(self):
        """Toggle push-to-talk from tray."""
        if self.server.engine.is_recording:
            self.server.engine.stop_listening()
        else:
            self.server.engine.start_listening()

    def stop_speaking(self):
        """Stop current speech playback."""
        self.server.engine.stop_speaking()

    def clear_history(self):
        """Clear conversation history."""
        self.server.engine.clear_history()

    def quit(self):
        """Gracefully terminate background server and tray icon."""
        self._stop_event.set()
        if self.server._listener:
            try:
                self.server._listener.stop()
            except Exception:
                pass
        self.server.engine.player.close()

        if self._server_loop and self._server_loop.is_running():
            self._server_loop.call_soon_threadsafe(self._server_loop.stop)

        if self.icon:
            self.icon.stop()

    def _run_server_thread(self):
        """Run aiohttp server in dedicated background thread."""
        self._server_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._server_loop)
        try:
            self._server_loop.run_until_complete(self.server.start())
            self._server_loop.run_forever()
        finally:
            self._server_loop.close()

    def run(self):
        """Start background server thread and launch Windows System Tray loop."""
        # 1. Start server in background thread
        self._server_thread = threading.Thread(target=self._run_server_thread, daemon=True)
        self._server_thread.start()

        # 2. Build system tray menu
        menu = Menu(
            item("JARPIS Voice Assistant", None, enabled=False),
            item(f"Port: {self.server.port}", None, enabled=False),
            Menu.SEPARATOR,
            item("Open Holographic HUD", lambda icon, item: self.open_hud(), default=True),
            item(
                lambda item: f"Status: {self.server.engine.status.capitalize()}",
                None,
                enabled=False,
            ),
            item("Toggle Voice (F8)", lambda icon, item: self.trigger_voice()),
            item("Stop Speaking", lambda icon, item: self.stop_speaking()),
            item("Reset Conversation", lambda icon, item: self.clear_history()),
            Menu.SEPARATOR,
            item("Exit JARPIS", lambda icon, item: self.quit()),
        )

        initial_icon = create_arc_reactor_icon(64, "#00e5ff")
        self.icon = pystray.Icon(
            name="JARPIS",
            icon=initial_icon,
            title=f"JARPIS Voice Assistant (Port {self.server.port})",
            menu=menu,
        )

        print("=" * 60)
        print(" JARPIS System Tray Background Daemon Active")
        print(f" - Local HUD: http://{self.server.host}:{self.server.port}")
        print(f" - WebSocket: ws://{self.server.host}:{self.server.port}/ws")
        print(f" - Push-To-Talk: Hold [{self.server.engine.target_key.upper()}] or tap [ENTER]")
        print(" - Minimizing to Windows Notification Area (System Tray)...")
        print("=" * 60)

        # Automatically pop open HUD on initial launch
        self.open_hud()

        # 3. Enter pystray main loop
        self.icon.run()


def main():
    app = JarpisTrayApp()
    app.run()


if __name__ == "__main__":
    main()
