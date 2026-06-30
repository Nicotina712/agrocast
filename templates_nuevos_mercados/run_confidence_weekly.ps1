# Weekly confidence-analyzer runner.
# Registered in Windows Task Scheduler (task: ConfidenceAnalyzer_Weekly).
# Runs confidence_analyzer.py with the same Python the robots use (has MT5)
# and writes a dated report into confidence_reports/.

$py      = "C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe"
$dir     = "C:\Users\Lenovo\OneDrive\Escritorio\MVP\templates_nuevos_mercados"
$reports = Join-Path $dir "confidence_reports"

New-Item -ItemType Directory -Force -Path $reports | Out-Null

$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$out   = Join-Path $reports "confidence_$stamp.txt"

Set-Location $dir
# Capture all streams and write UTF-8 (PowerShell's *> default is UTF-16).
& $py (Join-Path $dir "confidence_analyzer.py") 2>&1 | Out-File -FilePath $out -Encoding utf8
