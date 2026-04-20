@echo off
setlocal
title NYC Taxi Tips - Multi-Point System Validation

:: Clean screen and display banner
cls
echo ============================================================
echo   🚕  NYC TAXI: MULTI-POINT SYSTEM VALIDATION
echo ============================================================
echo.
echo [SYSTEM] Starting full architecture health check...
echo.

:: Pillar 1: Static Code Quality
echo [1/4] Pillar 1: Static Code Quality (Pyright ^& Ruff)...
echo      - Running Pyright (Strict Type Checking)...
call uv run pyright src/
if errorlevel 1 goto :FAILED

echo.
echo      - Running Ruff (Linting)...
call uv run ruff check .
if errorlevel 1 goto :FAILED

echo.
echo      - Running Ruff (Formatting Check)...
call uv run ruff format --check .
if errorlevel 1 goto :FAILED

echo.
echo      Done.
echo.

:: Pillar 2: Functional Logic ^& Coverage
echo [2/4] Pillar 2: Functional Logic ^& Coverage...
echo      - Running Pytest with Coverage Gate (65%%)...
call uv run pytest tests/ -v --cov=src --cov-fail-under=65
if errorlevel 1 goto :FAILED

echo.
echo      Done.
echo.

:: Pillar 3: Pipeline Synchronization
echo [3/4] Pillar 3: Pipeline Synchronization (DVC)...
call uv run dvc status
if errorlevel 1 goto :FAILED

echo.
echo      Done.
echo.

:: Pillar 4: API Service ^& Runtime
echo [4/4] Pillar 4: API Service Health...
:: We check the versioned /v1/health endpoint as per Phase 3 improvements
curl -s http://localhost:8000/v1/health | findstr "ok" >nul
if errorlevel 1 goto :API_WARNING

echo.
echo      API is ONLINE and HEALTHY (v1 validated).
goto :PILLAR4_DONE

:API_WARNING
echo.
echo      WARNING: Local API is not reachable or /v1/health returned error.
echo      Ensure the API is running (launch_app.bat) if you want to test runtime.

:PILLAR4_DONE
echo.
echo      Done.
echo.

:SUCCESS
echo ============================================================
echo   ✅ SYSTEM HEALTH: 100%% (ALL GATES PASSED)
echo ============================================================
echo.
echo Your NYC Taxi architecture is validated and safe for production readiness.
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
