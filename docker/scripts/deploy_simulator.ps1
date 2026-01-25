# Deploy Simulator - PowerShell version
# Usage: .\scripts\deploy_simulator.ps1

$ErrorActionPreference = "Stop"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  MARKOV RL API CACHE - SIMULATOR DEPLOYMENT" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "[OK] Docker is running" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Docker is not running. Please start Docker first." -ForegroundColor Red
    exit 1
}

# Check if docker-compose is available
try {
    docker-compose --version | Out-Null
    Write-Host "[OK] docker-compose found" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] docker-compose not found. Please install docker-compose first." -ForegroundColor Red
    exit 1
}

# Build images
Write-Host ""
Write-Host "[STEP 1] Building Docker images..." -ForegroundColor Yellow
docker-compose build --pull

# Start infrastructure
Write-Host ""
Write-Host "[STEP 2] Starting infrastructure services..." -ForegroundColor Yellow
docker-compose up -d redis prometheus grafana

Write-Host ""
Write-Host "[INFO] Waiting for infrastructure to be healthy..." -ForegroundColor Blue
Start-Sleep -Seconds 5

# Wait for Redis
Write-Host -NoNewline "  - Waiting for Redis... "
$timeout = 60
$elapsed = 0
while ($true) {
    try {
        docker-compose exec -T redis redis-cli ping | Out-Null
        Write-Host "OK" -ForegroundColor Green
        break
    } catch {
        Start-Sleep -Seconds 1
        $elapsed++
        if ($elapsed -ge $timeout) {
            Write-Host "TIMEOUT" -ForegroundColor Red
            exit 1
        }
    }
}

# Wait for Prometheus
Write-Host -NoNewline "  - Waiting for Prometheus... "
$elapsed = 0
while ($true) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "OK" -ForegroundColor Green
            break
        }
    } catch {
        Start-Sleep -Seconds 1
        $elapsed++
        if ($elapsed -ge $timeout) {
            Write-Host "TIMEOUT" -ForegroundColor Red
            exit 1
        }
    }
}

# Wait for Grafana
Write-Host -NoNewline "  - Waiting for Grafana... "
$elapsed = 0
while ($true) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "OK" -ForegroundColor Green
            break
        }
    } catch {
        Start-Sleep -Seconds 1
        $elapsed++
        if ($elapsed -ge $timeout) {
            Write-Host "TIMEOUT" -ForegroundColor Red
            exit 1
        }
    }
}

# Start microservices
Write-Host ""
Write-Host "[STEP 3] Starting microservices..." -ForegroundColor Yellow
docker-compose up -d auth-service user-service inventory-service product-service cart-service payment-service order-service

Write-Host ""
Write-Host "[INFO] Waiting for services to be healthy..." -ForegroundColor Blue
Start-Sleep -Seconds 10

# Check service health
function Test-ServiceHealth {
    param($ServiceName, $Port)

    Write-Host -NoNewline "  - Checking ${ServiceName}... "

    for ($i = 0; $i -lt 30; $i++) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:${Port}/health" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                Write-Host "OK" -ForegroundColor Green
                return $true
            }
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    Write-Host "TIMEOUT" -ForegroundColor Red
    return $false
}

# Check all services
$failedServices = @()

if (-not (Test-ServiceHealth "auth-service" 8002)) { $failedServices += "auth-service" }
if (-not (Test-ServiceHealth "user-service" 8001)) { $failedServices += "user-service" }
if (-not (Test-ServiceHealth "inventory-service" 8007)) { $failedServices += "inventory-service" }
if (-not (Test-ServiceHealth "product-service" 8003)) { $failedServices += "product-service" }
if (-not (Test-ServiceHealth "cart-service" 8004)) { $failedServices += "cart-service" }
if (-not (Test-ServiceHealth "payment-service" 8006)) { $failedServices += "payment-service" }
if (-not (Test-ServiceHealth "order-service" 8005)) { $failedServices += "order-service" }

Write-Host ""
if ($failedServices.Count -eq 0) {
    Write-Host "[SUCCESS] All services are healthy!" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Some services failed to start:" -ForegroundColor Yellow
    foreach ($service in $failedServices) {
        Write-Host "  - $service" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Run '.\scripts\logs.ps1' to see logs"
}

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  DEPLOYMENT COMPLETE" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Services Available:" -ForegroundColor Green
Write-Host "  - Auth Service:      http://localhost:8002"
Write-Host "  - User Service:      http://localhost:8001"
Write-Host "  - Product Service:   http://localhost:8003"
Write-Host "  - Cart Service:      http://localhost:8004"
Write-Host "  - Order Service:     http://localhost:8005"
Write-Host "  - Payment Service:   http://localhost:8006"
Write-Host "  - Inventory Service: http://localhost:8007"
Write-Host ""
Write-Host "Monitoring:" -ForegroundColor Green
Write-Host "  - Prometheus:        http://localhost:9090"
Write-Host "  - Grafana:           http://localhost:3000 (admin/admin)"
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Blue
Write-Host "  1. Check status:     docker-compose ps"
Write-Host "  2. Start traffic:    .\scripts\start_traffic.ps1 normal"
Write-Host "  3. View logs:        .\scripts\logs.ps1 [service-name]"
Write-Host "  4. Stop all:         .\scripts\stop_all.ps1"
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan

