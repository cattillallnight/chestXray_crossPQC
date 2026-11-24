$ErrorActionPreference = "Stop"
$python = "C:\Users\Thuy Quyen\AppData\Local\Programs\Python\Python310\python.exe"
$root = "E:\cvchestpj"
$outLog = Join-Path $root "results\overnight_stdout.log"
$errLog = Join-Path $root "results\overnight_stderr.log"
$pidFile = Join-Path $root "results\overnight.pid"

New-Item -ItemType Directory -Force -Path (Join-Path $root "results") | Out-Null

# Stop the chat-attached queue so we do not double-train the same run.
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and (
        $_.CommandLine -match "run_queue\.py" -or
        $_.CommandLine -match "scripts\\train\.py"
    )
} | ForEach-Object {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
}
Start-Sleep -Seconds 2

# Detached process: survives this PowerShell and the Raycast chat.
# Windows forbids redirecting stdout and stderr to the same file.
$p = Start-Process -FilePath $python -ArgumentList @("-u", "$root\scripts\run_queue.py") `
    -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $outLog `
    -RedirectStandardError $errLog -PassThru

$p.Id | Set-Content -Path $pidFile -Encoding ascii
"launched pid=$($p.Id) stdout=$outLog stderr=$errLog"
