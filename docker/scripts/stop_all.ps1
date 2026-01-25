# Stop All Services - PowerShell version
# Usage: .\scripts\stop_all.ps1 [-Clean]

param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  STOPPING ALL SERVICES" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Stop traffic generator
Write-Host "[INFO] Stopping traffic generator..." -ForegroundColor Blue
docker-compose --profile traffic stop traffic-generator 2>$null
docker-compose --profile traffic rm -f traffic-generator 2>$null
Write-Host "[OK] Traffic generator stopped" -ForegroundColor Green

# Stop application services
Write-Host ""
Write-Host "[INFO] Stopping application services..." -ForegroundColor Blue
docker-compose stop order-service payment-service cart-service product-service inventory-service user-service auth-service
Write-Host "[OK] Application services stopped" -ForegroundColor Green

# Stop infrastructure
Write-Host ""
Write-Host "[INFO] Stopping infrastructure services..." -ForegroundColor Blue
docker-compose stop grafana prometheus redis
Write-Host "[OK] Infrastructure services stopped" -ForegroundColor Green

# Remove containers
Write-Host ""
Write-Host "[INFO] Removing containers..." -ForegroundColor Blue
docker-compose down
Write-Host "[OK] Containers removed" -ForegroundColor Green

# Clean volumes if requested
if ($Clean) {
    Write-Host ""
    Write-Host "[WARNING] Removing volumes (this will delete all data)..." -ForegroundColor Yellow
    $confirmation = Read-Host "Are you sure? (yes/no)"
    if ($confirmation -eq "yes") {
        docker-compose down -v
        Write-Host "[OK] Volumes removed" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Volume removal cancelled" -ForegroundColor Blue
    }
}

# Show remaining resources
Write-Host ""
Write-Host "[INFO] Checking for remaining resources..." -ForegroundColor Blue

$containers = docker ps -a --filter "name=simulator-" --format "{{.Names}}"
$volumes = docker volume ls --filter "name=markov-rl-api-cache" --format "{{.Name}}"

if ($containers) {
    Write-Host "[WARNING] Some containers still exist:" -ForegroundColor Yellow
    foreach ($container in $containers) {
        Write-Host "  - $container"
    }
}

if ($volumes) {
    Write-Host "[INFO] Persistent volumes still exist:" -ForegroundColor Blue
    foreach ($volume in $volumes) {
        Write-Host "  - $volume"
    }
    Write-Host ""
    Write-Host "To remove volumes: .\scripts\stop_all.ps1 -Clean"
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] All services stopped" -ForegroundColor Green
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start again: .\scripts\deploy_simulator.ps1"
Write-Host ""

