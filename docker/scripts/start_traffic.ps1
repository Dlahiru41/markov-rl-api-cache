# Start Traffic Generator - PowerShell version
# Usage: .\scripts\start_traffic.ps1 [profile]
# Profiles: normal, peak, degraded, burst

param(
    [string]$Profile = "normal"
)

$ErrorActionPreference = "Stop"

$validProfiles = @("normal", "peak", "degraded", "burst")

if ($validProfiles -notcontains $Profile) {
    Write-Host "[ERROR] Invalid profile: $Profile" -ForegroundColor Red
    Write-Host "Valid profiles: $($validProfiles -join ', ')"
    exit 1
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  TRAFFIC GENERATOR - Starting with profile: $Profile" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if services are running
Write-Host "[INFO] Checking if services are running..." -ForegroundColor Blue
$services = docker-compose ps
if ($services -notmatch "Up") {
    Write-Host "[ERROR] Services are not running. Please run '.\scripts\deploy_simulator.ps1' first." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Services are running" -ForegroundColor Green

# Stop existing traffic generator
Write-Host ""
Write-Host "[INFO] Stopping any existing traffic generator..." -ForegroundColor Blue
docker-compose --profile traffic stop traffic-generator 2>$null
docker-compose --profile traffic rm -f traffic-generator 2>$null

# Start traffic generator
Write-Host ""
Write-Host "[STARTING] Traffic generator with profile: $Profile" -ForegroundColor Yellow
$env:TRAFFIC_PROFILE = $Profile
docker-compose --profile traffic up -d traffic-generator

Write-Host ""
Write-Host "[INFO] Traffic generator started. Monitoring logs..." -ForegroundColor Blue
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Follow logs
docker-compose --profile traffic logs -f traffic-generator

