<#
.SYNOPSIS
    Register the OCR platform and its nightly backup as Windows scheduled
    tasks, so the service survives a reboot and a crash.

.DESCRIPTION
    Uses Task Scheduler rather than NSSM because it needs no download and no
    admin-installed third-party binary. The service task starts at boot and
    restarts on failure; the backup task runs nightly.

    Run this from an ELEVATED PowerShell prompt:

        powershell -ExecutionPolicy Bypass -File tools\install_windows_service.ps1

    Remove them again with:

        Unregister-ScheduledTask -TaskName "OCRPlatform"       -Confirm:$false
        Unregister-ScheduledTask -TaskName "OCRPlatformBackup" -Confirm:$false
#>

param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ServiceTask = "OCRPlatform",
    [string]$BackupTask  = "OCRPlatformBackup",
    [string]$BackupAt    = "02:30"
)

$ErrorActionPreference = "Stop"

if (-not ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this from an elevated PowerShell prompt."
}

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Interpreter not found: $python" }

Write-Host "Project : $ProjectRoot"
Write-Host "Python  : $python`n"

# --- the service ---------------------------------------------------------
# RestartCount/RestartInterval are what turn "it crashed" into "it came back".
$action  = New-ScheduledTaskAction -Execute $python -Argument "run.py" -WorkingDirectory $ProjectRoot
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask -TaskName $ServiceTask -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -Force `
    -Description "AI Document Intelligence Platform (FastAPI/uvicorn)" | Out-Null
Write-Host "Registered '$ServiceTask' - starts at boot, restarts up to 5 times on failure."

# --- nightly backup ------------------------------------------------------
$bAction  = New-ScheduledTaskAction -Execute $python `
            -Argument "tools\backup_db.py --keep 14" -WorkingDirectory $ProjectRoot
$bTrigger = New-ScheduledTaskTrigger -Daily -At $BackupAt
$bSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName $BackupTask -Action $bAction -Trigger $bTrigger `
    -Settings $bSettings -Principal $principal -Force `
    -Description "Nightly database backup, 14 kept" | Out-Null
Write-Host "Registered '$BackupTask' - runs daily at $BackupAt, keeps 14 backups.`n"

Write-Host "Start it now with:  Start-ScheduledTask -TaskName $ServiceTask"
Write-Host "Check health with:  curl http://127.0.0.1:8080/health"
Write-Host ""
Write-Host "NOTE: this serves plain HTTP. The session cookie is marked Secure in"
Write-Host "production and will NOT travel over http, so put a TLS-terminating"
Write-Host "reverse proxy (IIS, Caddy, nginx) in front before exposing the host."
