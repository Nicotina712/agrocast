@echo off
REM BTCUSD Bitcoin Robot â€” Launcher
REM Puerto dashboard: 8081 (Gold usa 8080)
REM NOTA: EXECUTE_TRADES=False en config.py â€” activar manualmente despues de paper trading

set py -3=C:\Users\Lenovo\AppData\Local\Programs\py -3\Python312\py -3.exe
set RUNNER=%~dp0live_runner.py
set PID_FILE=%~dp0..\..\artifacts\btcusd\robot.pid

echo ============================================
echo  BTCUSD Bitcoin Robot
echo  Prime session: 07:00-22:00 CT (24/7)
echo  Dashboard: http://localhost:8081
echo ============================================
echo.

if "%1"=="--loop" (
    echo Iniciando loop paper trading BTC...
    "%py -3%" "%RUNNER%" --loop
) else if "%1"=="--execute" (
    echo ATENCION: Ejecucion en vivo en cuenta DEMO activada
    start /min "" "%py -3%" "%RUNNER%" --loop --execute
    echo Robot BTC iniciado en background
    echo PID guardado en: %PID_FILE%
) else if "%1"=="--diagnose" (
    "%py -3%" "%RUNNER%" --diagnose
) else if "%1"=="--report" (
    "%py -3%" "%~dp0execution_tracker.py" --report
) else if "%1"=="--retrain" (
    "%py -3%" "%~dp0retrainer.py"
) else if "%1"=="--dashboard" (
    start "" "http://localhost:8081"
    "%py -3%" "%~dp0dashboard.py"
) else if "%1"=="--status" (
    if exist "%PID_FILE%" (
        set /p PID=<"%PID_FILE%"
        echo PID del robot BTC: !PID!
        tasklist /FI "PID eq !PID!" 2>NUL | find "!PID!" >NUL
        if errorlevel 1 (echo Estado: DETENIDO) else (echo Estado: CORRIENDO)
    ) else (
        echo No hay robot.pid guardado para BTC
    )
    echo.
    echo Ultimas 5 entradas del log:
    "%py -3%" -c "import json; lines=open(r'%~dp0..\..\artifacts\btcusd\live_log.jsonl').readlines()[-5:]; [print(json.loads(l)['ct_time'], json.loads(l)['type'], json.loads(l).get('price','')) for l in lines]" 2>NUL
) else if "%1"=="--stop" (
    if exist "%PID_FILE%" (
        set /p PID=<"%PID_FILE%"
        echo Deteniendo robot BTC PID !PID!...
        taskkill /PID !PID! /F
        del "%PID_FILE%"
        echo Robot BTC detenido.
    ) else (
        echo No hay PID guardado para BTC.
    )
) else (
    echo Uso:
    echo   run_btc.bat --diagnose    Verificar conexion MT5 + simbolo BTCUSD
    echo   run_btc.bat --loop        Solo paper trading (EXECUTE_TRADES=False)
    echo   run_btc.bat --execute     Iniciar robot con ejecucion real (demo)
    echo   run_btc.bat --status      Ver si el robot esta corriendo
    echo   run_btc.bat --stop        Detener el robot
    echo   run_btc.bat --report      Ver metricas de paper trades
    echo   run_btc.bat --retrain     Reentrenar modelo (domingos)
    echo   run_btc.bat --dashboard   Abrir dashboard en puerto 8081
    echo.
    echo NOTA: Config activa = paper trading (EXECUTE_TRADES=False)
    echo Activar --execute solo despues de 2 semanas de paper trading positivo.
    echo.
    echo Corriendo un ciclo de prueba...
    "%py -3%" "%RUNNER%"
)

