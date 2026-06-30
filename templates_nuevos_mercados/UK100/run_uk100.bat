@echo off
REM UK100 Robot Launcher
cd /d "%~dp0"

set CMD=%1
if "%CMD%"=="" set CMD=--report

if "%CMD%"=="--loop"      goto loop
if "%CMD%"=="--execute"   goto execute
if "%CMD%"=="--diagnose"  goto diagnose
if "%CMD%"=="--report"    goto report
if "%CMD%"=="--retrain"   goto retrain
if "%CMD%"=="--dashboard" goto dashboard
if "%CMD%"=="--stop"      goto stop

:usage
echo Usage: run_uk100.bat [--loop^|--execute^|--diagnose^|--report^|--retrain^|--dashboard^|--stop]
goto end

:loop
echo Starting UK100 robot (paper mode, loop)...
start "UK100_robot" py -3 live_runner.py --loop
goto end

:execute
echo Starting UK100 robot (LIVE EXECUTION)...
start "UK100_robot" py -3 live_runner.py --loop --execute
goto end

:diagnose
py -3 live_runner.py --diagnose
goto end

:report
py -3 execution_tracker.py --report
goto end

:retrain
py -3 retrainer.py
goto end

:dashboard
echo Starting UK100 dashboard on port 8088...
start "UK100_dash" py -3 dashboard.py
timeout /t 2 >nul
start http://localhost:8088
goto end

:stop
echo Stopping UK100 robot...
taskkill /FI "WINDOWTITLE eq UK100_robot" /F 2>nul
taskkill /FI "WINDOWTITLE eq UK100_dash"  /F 2>nul
echo Done.
goto end

:end

