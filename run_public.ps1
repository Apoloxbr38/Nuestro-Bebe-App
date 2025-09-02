# Guarda esto como C:\Proyectos\sports-predictor\run_public.ps1
param(
  [string]$VenvPath = "C:\Proyectos\sports-predictor\.venv311"
)

$ErrorActionPreference = "Stop"
Set-Location "C:\Proyectos\sports-predictor"

# Activar venv
& "$VenvPath\Scripts\Activate.ps1"

# Instalar deps (idempotente)
pip install -r backend\requirements.txt | Out-Null

# Arrancar backend en nueva ventana
Start-Process -WindowStyle Normal powershell -ArgumentList "-NoExit","-Command","python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload"

Start-Sleep -Seconds 2

# Lanzar ngrok y capturar salida
$ngrok = Start-Process -PassThru -WindowStyle Hidden -FilePath "ngrok" -ArgumentList "http 8000" -RedirectStandardOutput "ngrok.log" -RedirectStandardError "ngrok.err"
Start-Sleep -Seconds 2

# Leer la URL pública del log (muy simple)
$publicUrl = $null
$deadline = (Get-Date).AddSeconds(15)
while (-not $publicUrl -and (Get-Date) -lt $deadline) {
  Start-Sleep -Milliseconds 500
  if (Test-Path "ngrok.log") {
    $line = Get-Content "ngrok.log" -Raw
    if ($line -match "https://[a-z0-9\-]+\.ngrok-free\.app") {
      $publicUrl = $Matches[0]
    }
  }
}

if ($publicUrl) {
  $full = "$publicUrl/app/"
  Write-Host "URL pública: $full"
  Start-Process $full
} else {
  Write-Warning "No pude detectar la URL de ngrok. Ábrela en su ventana."
  Start-Process "http://127.0.0.1:8000/app/"
}
