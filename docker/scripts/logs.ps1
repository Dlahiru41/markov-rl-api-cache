# Logs Viewer - PowerShell version
# Usage: .\scripts\logs.ps1 [-Service <name>] [-Level <level>] [-NoFollow]

param(
    [string]$Service = "",
    [string]$Level = "",
    [switch]$NoFollow
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  LOGS VIEWER" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

if ($Service -eq "") {
    Write-Host "[INFO] Showing logs from all services" -ForegroundColor Blue
    Write-Host ""

    if ($NoFollow) {
        docker-compose logs --tail=50
    } else {
        docker-compose logs -f --tail=50
    }
} else {
    # Check if service exists
    $services = docker-compose ps --services
    if ($services -notcontains $Service) {
        Write-Host "[ERROR] Service not found: $Service" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available services:"
        foreach ($svc in $services) {
            Write-Host "  - $svc"
        }
        exit 1
    }

    Write-Host "[INFO] Showing logs from: $Service" -ForegroundColor Blue
    Write-Host ""

    if ($NoFollow) {
        if ($Level -ne "") {
            docker-compose logs --tail=50 $Service | Select-String -Pattern $Level
        } else {
            docker-compose logs --tail=50 $Service
        }
    } else {
        if ($Level -ne "") {
            docker-compose logs -f --tail=50 $Service | Select-String -Pattern $Level
        } else {
            docker-compose logs -f --tail=50 $Service
        }
    }
}

