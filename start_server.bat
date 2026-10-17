@echo off
title JARPIS - Holographic HUD Server
cd /d "%~dp0"
echo ============================================================
echo  Starting JARPIS Holographic HUD Server (Port 20130)
echo ============================================================
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -u run_server.py
) else (
    python -u run_server.py
)
pause
