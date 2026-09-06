@echo off
title TriagePulse AI - Clinical & Latency Benchmark Suite
echo =======================================================
echo   TriagePulse AI - Comprehensive Evaluation Suite
echo =======================================================
call .\venv\Scripts\activate.bat

echo [1/3] Running 100+ Emergency Safety Tests (0%% FN target)...
pytest eval/test_red_flags.py -v --tb=short
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Red Flag safety evaluation failed!
    pause
    exit /b 1
)

echo [2/3] Running End-to-End & Unit Tests (131 cases)...
pytest tests/ -v --tb=short
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Test suite failed!
    pause
    exit /b 1
)

echo [3/3] Running Turn-by-Turn Latency Waterfall Benchmark (50 turns)...
python eval/test_latency.py

echo.
echo =======================================================
echo   All Evaluations & Benchmarks Completed Successfully!
echo =======================================================
pause
