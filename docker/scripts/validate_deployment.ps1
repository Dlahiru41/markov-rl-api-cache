# Validation Script - PowerShell version
# Usage: .\scripts\validate_deployment.ps1

$ErrorActionPreference = "Continue"

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  DOCKER DEPLOYMENT VALIDATION" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

$failed = 0

# Test 1: Docker
Write-Host "[TEST 1] Checking Docker..." -ForegroundColor Blue
try {
    docker info | Out-Null
    Write-Host "[OK] Docker is running" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Docker is not running" -ForegroundColor Red
    exit 1
}

# Test 2: docker-compose
Write-Host ""
Write-Host "[TEST 2] Checking docker-compose..." -ForegroundColor Blue
try {
    docker-compose --version | Out-Null
    Write-Host "[OK] docker-compose is available" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] docker-compose not found" -ForegroundColor Red
    exit 1
}

# Test 3: Services running
Write-Host ""
Write-Host "[TEST 3] Checking if services are running..." -ForegroundColor Blue
$services = docker-compose ps
if ($services -match "Up") {
    Write-Host "[OK] Services are running" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Services not running. Deploy with: .\scripts\deploy_simulator.ps1" -ForegroundColor Yellow
    exit 0
}

# Test 4: Health checks
Write-Host ""
Write-Host "[TEST 4] Checking service health..." -ForegroundColor Blue

function Test-Health {
    param($Name, $Port)

    try {
        $response = Invoke-WebRequest -Uri "http://localhost:${Port}/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "  ${Name}: [HEALTHY]" -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host "  ${Name}: [UNHEALTHY]" -ForegroundColor Red
        return $false
    }
}

if (-not (Test-Health "auth-service" 8002)) { $failed++ }
if (-not (Test-Health "user-service" 8001)) { $failed++ }
if (-not (Test-Health "product-service" 8003)) { $failed++ }
if (-not (Test-Health "cart-service" 8004)) { $failed++ }
if (-not (Test-Health "order-service" 8005)) { $failed++ }
if (-not (Test-Health "payment-service" 8006)) { $failed++ }
if (-not (Test-Health "inventory-service" 8007)) { $failed++ }

# Test 5: Prometheus
Write-Host ""
Write-Host "[TEST 5] Checking Prometheus..." -ForegroundColor Blue
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 2
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] Prometheus is healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "[FAIL] Prometheus is not accessible" -ForegroundColor Red
    $failed++
}

# Test 6: Grafana
Write-Host ""
Write-Host "[TEST 6] Checking Grafana..." -ForegroundColor Blue
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 2
    if ($response.StatusCode -eq 200) {
        Write-Host "[OK] Grafana is healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "[FAIL] Grafana is not accessible" -ForegroundColor Red
    $failed++
}

# Test 7: Redis
Write-Host ""
Write-Host "[TEST 7] Checking Redis..." -ForegroundColor Blue
try {
    docker-compose exec -T redis redis-cli ping | Out-Null
    Write-Host "[OK] Redis is responding" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Redis is not responding" -ForegroundColor Red
    $failed++
}

# Summary
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
if ($failed -eq 0) {
    Write-Host "[SUCCESS] All validation tests passed!" -ForegroundColor Green
} else {
    Write-Host "[WARNING] $failed test(s) failed" -ForegroundColor Yellow
}
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

if ($failed -eq 0) {
    Write-Host "Deployment is healthy!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access points:"
    Write-Host "  - Prometheus: http://localhost:9090"
    Write-Host "  - Grafana:    http://localhost:3000 (admin/admin)"
    Write-Host ""
    Write-Host "Next: Start traffic with '.\scripts\start_traffic.ps1 normal'"
}

exit $failed

