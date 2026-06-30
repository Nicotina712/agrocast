@echo off
REM Portfolio News Intelligence - Hourly Runner
REM Registrar en Task Scheduler para correr cada hora
REM Accion: python run_news_pipeline.py
REM Trigger: cada 1 hora, todos los dias

cd /d "%~dp0"
cd ..\..

echo [%date% %time%] Iniciando news pipeline...
py -3 templates_nuevos_mercados\news_intelligence\run_news_pipeline.py >> logs\news_pipeline.log 2>&1
echo [%date% %time%] Pipeline finalizado.
