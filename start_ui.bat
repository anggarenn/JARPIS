@echo off
title JARPIS - Stark Industries Holographic HUD
cd /d "%~dp0"
echo ============================================================
echo   J.A.R.V.I.S. - STARK INDUSTRIES SECOND SCREEN HUD
echo   Server Port : 20130 (Local WebSocket & Web Telemetry)
echo   Local URL   : http://127.0.0.1:20130
echo ============================================================

REM Launch default browser pointing to the HUD after a short delay
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:20130"

REM Run the async web & WebSocket server
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -u run_server.py
) else (
    python -u run_server.py
)
pause
