@echo off
setlocal
title NYC Taxi Tips Predictor - Launcher

:: Clean screen and display banner
cls
echo ============================================================
echo   🚕 NYC TAXI TIPS PREDICTOR: AGENTIC MLOPS SYSTEM
echo ============================================================
echo.
echo [SYSTEM] Initializing Antigravity Stack...
echo.

:: Step 1: Check/Sync Dependencies
echo [1/3] Verifying dependencies with UV...
uv sync --quiet
if "%ERRORLEVEL%" NEQ "0" (
    echo.
    echo 🚨 Error: Failed to sync dependencies. Verify 'uv' is installed.
    pause
    exit /b %ERRORLEVEL%
)
echo      Done.
echo.

:: Step 2: Launch FastAPI in a separate minimized window
echo [2/3] Launching Inference API (FastAPI)...
echo      Endpoint: http://localhost:8000
:: Start the API window minimized to keep it tidy but accessible
start "NYC-TAXI-API" /min cmd /k "title NYC-TAXI-API && uv run uvicorn src.api.predict_api:app --host 0.0.0.0 --port 8000 --reload"

:: Wait for API to warm up
echo.
echo [WAIT] Stalling for API initialization (5s)...
timeout /t 5 >nul

:: Step 3: Launch Streamlit in the foreground
echo.
echo [3/3] Launching Frontend Dashboard (Streamlit)...
echo      URL: http://localhost:8501
echo.
echo ------------------------------------------------------------
echo 💡 TIP: The API is running in the background (minimized).
echo    To stop EVERYTHING:
echo    1. Close the "NYC-TAXI-API" window in the taskbar.
echo    2. Press Ctrl+C in this window.
echo ------------------------------------------------------------
echo.

:: Run Streamlit
uv run streamlit run src/app/main.py

:: If the user stops Streamlit, give them a chance to read the exit message
echo.
echo [SYSTEM] Application Sessions Terminated.
pause
