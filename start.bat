@echo off
title JARPIS - Jarvis Voice Assistant
cd /d "%~dp0"
echo ============================================================
echo  Starting JARPIS Voice Assistant (Sub-800ms Latency)
echo ============================================================
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -u run.py
) else (
    python -u run.py
)
pause
