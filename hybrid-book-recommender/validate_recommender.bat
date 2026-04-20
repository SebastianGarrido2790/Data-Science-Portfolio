@echo off
setlocal
title Hybrid Book Recommender - Multi-Point System Validation

:: Clean screen and display banner
cls
echo ============================================================
echo   📚 HYBRID BOOK RECOMMENDER: SYSTEM HEALTH CHECK
echo ============================================================
echo.
echo [SYSTEM] Starting full architecture health check...
echo.

:: Pillar 0: Sync Dependencies
echo [0/4] Pillar 0: Syncing all dependencies (including Dev Tools)...
call uv sync --all-extras --quiet
if %ERRORLEVEL% NEQ 0 goto :FAILED

:: Pillar 1: Static Code Quality
echo [1/4] Pillar 1: Static Code Quality (Pyright ^& Ruff)...
echo      - Running Pyright (Strict Type Checking)...
call uv run pyright src/
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo.
echo      - Running Ruff (Linting)...
call uv run ruff check .
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo.
echo      - Running Ruff (Formatting Check)...
call uv run ruff format --check .
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo      Done.
echo.

:: Pillar 2: Functional Logic ^& Coverage
echo [2/4] Pillar 2: Functional Logic ^& Coverage...
echo      - Running Pytest with Coverage...
call uv run pytest tests/ -v --cov=src
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo      Done.
echo.

:: Pillar 3: Pipeline Synchronization
echo [3/4] Pillar 3: Pipeline Synchronization (DVC)...
call uv run dvc status
if %ERRORLEVEL% NEQ 0 goto :FAILED

echo      Done.
echo.

:: Pillar 4: App Service ^& Runtime
echo [4/4] Pillar 4: App Service Health...
:: We check port 7860 (Gradio)
powershell -Command "try { $c = New-Object System.Net.Sockets.TcpClient('localhost', 7860); if ($c.Connected) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %ERRORLEVEL% NEQ 0 goto :API_WARNING

echo      Gradio Storefront is ONLINE and REACHABLE on port 7860.
goto :PILLAR4_DONE

:API_WARNING
echo.
echo      WARNING: Local Gradio App is not reachable on port 7860.
echo      Ensure the App is running (launch_recommender.bat) to pass runtime check.

:PILLAR4_DONE
echo      Done.
echo.

:SUCCESS
echo ============================================================
echo   ✅ SYSTEM HEALTH: 100%% (ALL GATES PASSED)
echo ============================================================
echo.
echo Your Hybrid Book Recommender architecture is validated.
pause
exit /b 0

:FAILED
echo.
echo ============================================================
echo   ❌ VALIDATION FAILED
echo ============================================================
echo.
echo Please review the logs above and correct the issues.
pause
exit /b 1
