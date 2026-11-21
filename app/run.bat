@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe ( echo Please run install.bat first. & pause & exit /b 1 )
echo BirdCull is starting... your browser will open. Keep this window open while you use it.
.venv\Scripts\python.exe run_birdcull.py
pause
