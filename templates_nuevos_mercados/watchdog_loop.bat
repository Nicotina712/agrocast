@echo off
title Portfolio Watchdog — Auto-Restart
cd /d "%~dp0"

echo ============================================
echo  Portfolio Watchdog Loop — iniciando
echo  Verifica robots y dashboards cada 5 min
echo ============================================
echo.

:loop
py -3 watchdog.py
echo.
echo [%time%] Proxima revision en 5 minutos...
timeout /t 300 /nobreak >nul
goto loop
