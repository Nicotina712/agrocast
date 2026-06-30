@echo off
REM XAUUSD Gold Robot â€” Launcher
REM Usa el py -3 que tiene MetaTrader5 instalado

set py -3=C:\Users\Lenovo\AppData\Local\Programs\py -3\Python312\py -3.exe
set RUNNER=%~dp0live_runner.py
set PID_FILE=%~dp0..\..\artifacts\xauusd\robot.pid

echo ============================================
echo  XAUUSD Gold Robot
echo  RTH: 07:00-11:30 CT (08:00-12:30 ET)
echo ============================================
echo.

if "%1"=="--loop" (
    echo Iniciando loop paper trading...
    "%py -3%" "%RUNNER%" --loop
) else if "%1"=="--execute" (
    echo ATENCION: Ejecucion en vivo en cuenta DEMO activada
    start /min "" "%py -3%" "%RUNNER%" --loop --execute
    echo Robot iniciado en background
) else if "%1"=="--diagnose" (
    "%py -3%" "%RUNNER%" --diagnose
) else if "%1"=="--report" (
    "%py -3%" "%~dp0execution_tracker.py" --report
) else if "%1"=="--retrain" (
    "%py -3%" "%~dp0retrainer.py"
) else if "%1"=="--dashboard" (
    start "" "http://localhost:8080"
    "%py -3%" "%~dp0dashboard.py"
) else if "%1"=="--status" (
    if exist "%PID_FILE%" (
        set /p PID=<"%PID_FILE%"
        echo PID del robot: !PID!
        tasklist /FI "PID eq !PID!" 2>NUL | find "!PID!" >NUL
        if errorlevel 1 (echo Estado: DETENIDO) else (echo Estado: CORRIENDO)
    ) else (
        echo No hay robot.pid guardado
    )
    echo.
    echo Ultimas 5 entradas del log:
    "%py -3%" -c "import json; lines=open(r'%~dp0..\..\artifacts\xauusd\live_log.jsonl').readlines()[-5:]; [print(json.loads(l)['ct_time'], json.loads(l)['type'], json.loads(l).get('price','')) for l in lines]" 2>NUL
) else if "%1"=="--stop" (
    if exist "%PID_FILE%" (
        set /p PID=<"%PID_FILE%"
        echo Deteniendo robot PID !PID!...
        taskkill /PID !PID! /F
        del "%PID_FILE%"
        echo Robot detenido.
    ) else (
        echo No hay PID guardado. Busca el proceso py -3 manualmente.
    )
) else (
    echo Uso:
    echo   run_gold.bat --diagnose    Verificar conexion MT5
    echo   run_gold.bat --execute     Iniciar robot con ejecucion real (demo)  ^<-- ACTIVO
    echo   run_gold.bat --loop        Solo paper trading (sin ejecutar)
    echo   run_gold.bat --status      Ver si el robot esta corriendo
    echo   run_gold.bat --stop        Detener el robot
    echo   run_gold.bat --report      Ver metricas de trades
    echo   run_gold.bat --retrain     Reentrenar modelo (domingos)
    echo.
    echo Corriendo un ciclo de prueba...
    "%py -3%" "%RUNNER%"
)

