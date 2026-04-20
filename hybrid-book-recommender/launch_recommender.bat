@echo off
setlocal
title Hybrid Book Recommender - Launch Suite

:: Clean screen and display banner
cls
echo ============================================================
echo   📚 HYBRID BOOK RECOMMENDER: AGENTIC STOREFRONT
echo ============================================================
echo.
echo [SYSTEM] Initializing Antigravity Stack...
echo.

:: Step 1: Check/Sync Dependencies
echo [1/2] Verifying dependencies with UV...
uv sync --quiet
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 🚨 Error: Failed to sync dependencies. Verify 'uv' is installed.
    pause
    exit /b %ERRORLEVEL%
)
echo      Done.
echo.

:: Step 2: Launch Gradio App
echo [2/2] Launching Intelligence Interface (Gradio)...
echo      URL: http://localhost:7860
echo.
echo ------------------------------------------------------------
echo 💡 TIP: Press Ctrl+C in this window to stop the server.
echo ------------------------------------------------------------
echo.

:: Run the modular package
uv run python -m src.app.main

:: Handle exit
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [SYSTEM] Application terminated unexpectedly.
) else (
    echo.
    echo [SYSTEM] Session ended successfully.
)

pause
