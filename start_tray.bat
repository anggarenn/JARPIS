@echo off
title JARPIS - System Tray Background Daemon
cd /d "%~dp0"
echo ============================================================
echo  Starting JARPIS in Windows System Tray (Background Mode)
echo ============================================================
if exist .venv\Scripts\pythonw.exe (
    start "" ".venv\Scripts\pythonw.exe" run_tray.py
    echo [JARPIS launched in background. Check your Windows System Tray icon!]
) else if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -u run_tray.py
) else (
    python -u run_tray.py
)
