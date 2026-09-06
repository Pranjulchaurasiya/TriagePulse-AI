@echo off
title TriagePulse AI - Production Console
echo =======================================================
echo   TriagePulse AI - Clinical Voice Reception & Triage
echo =======================================================
echo Activating virtual environment...
call .\venv\Scripts\activate.bat
echo Starting WebSocket & HTTP Gateway on http://127.0.0.1:8000 ...
python -m uvicorn perception.ws_gateway:app --host 127.0.0.1 --port 8000 --reload
pause
