@echo off
REM Master Portfolio Dashboard â€” All Instruments on port 8090
cd /d "%~dp0"

set CMD=%1
if "%CMD%"=="" set CMD=--start

if "%CMD%"=="--start" goto start
if "%CMD%"=="--stop"  goto stop

:start
echo.
echo ============================================
echo   Master Portfolio Dashboard
echo   http://localhost:8090
echo ============================================
echo.
start "MASTER_DASH" py -3 master_dashboard.py
timeout /t 2 >nul
start http://localhost:8090
goto end

:stop
echo Stopping master dashboard...
taskkill /FI "WINDOWTITLE eq MASTER_DASH" /F 2>nul
echo Done.
goto end

:end

