#!/bin/bash
# Validation Script - Test Docker deployment
# Usage: ./validate_deployment.sh

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================================================"
echo "  DOCKER DEPLOYMENT VALIDATION"
echo "================================================================================"
echo ""

# Test 1: Check Docker
echo -e "${BLUE}[TEST 1]${NC} Checking Docker..."
if docker info > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} Docker is running"
else
    echo -e "${RED}[FAIL]${NC} Docker is not running"
    exit 1
fi

# Test 2: Check docker-compose
echo ""
echo -e "${BLUE}[TEST 2]${NC} Checking docker-compose..."
if docker-compose --version > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} docker-compose is available"
else
    echo -e "${RED}[FAIL]${NC} docker-compose not found"
    exit 1
fi

# Test 3: Check services are running
echo ""
echo -e "${BLUE}[TEST 3]${NC} Checking if services are running..."
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}[OK]${NC} Services are running"
else
    echo -e "${YELLOW}[WARNING]${NC} Services not running. Deploy with: ./scripts/deploy_simulator.sh"
    exit 0
fi

# Test 4: Health checks
echo ""
echo -e "${BLUE}[TEST 4]${NC} Checking service health..."

check_health() {
    local service=$1
    local port=$2

    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo -e "  ${service}: ${GREEN}[HEALTHY]${NC}"
        return 0
    else
        echo -e "  ${service}: ${RED}[UNHEALTHY]${NC}"
        return 1
    fi
}

failed=0
check_health "auth-service" 8002 || failed=$((failed + 1))
check_health "user-service" 8001 || failed=$((failed + 1))
check_health "product-service" 8003 || failed=$((failed + 1))
check_health "cart-service" 8004 || failed=$((failed + 1))
check_health "order-service" 8005 || failed=$((failed + 1))
check_health "payment-service" 8006 || failed=$((failed + 1))
check_health "inventory-service" 8007 || failed=$((failed + 1))

# Test 5: Prometheus
echo ""
echo -e "${BLUE}[TEST 5]${NC} Checking Prometheus..."
if curl -sf "http://localhost:9090/-/healthy" > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} Prometheus is healthy"

    # Check targets
    targets=$(curl -s "http://localhost:9090/api/v1/targets" | grep -o '"health":"up"' | wc -l)
    echo -e "  Targets up: ${targets}"
else
    echo -e "${RED}[FAIL]${NC} Prometheus is not accessible"
    failed=$((failed + 1))
fi

# Test 6: Grafana
echo ""
echo -e "${BLUE}[TEST 6]${NC} Checking Grafana..."
if curl -sf "http://localhost:3000/api/health" > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} Grafana is healthy"
else
    echo -e "${RED}[FAIL]${NC} Grafana is not accessible"
    failed=$((failed + 1))
fi

# Test 7: Redis
echo ""
echo -e "${BLUE}[TEST 7]${NC} Checking Redis..."
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} Redis is responding"
else
    echo -e "${RED}[FAIL]${NC} Redis is not responding"
    failed=$((failed + 1))
fi

# Summary
echo ""
echo "================================================================================"
if [ $failed -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS]${NC} All validation tests passed!"
else
    echo -e "${YELLOW}[WARNING]${NC} ${failed} test(s) failed"
fi
echo "================================================================================"
echo ""

if [ $failed -eq 0 ]; then
    echo "Deployment is healthy!"
    echo ""
    echo "Access points:"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana:    http://localhost:3000 (admin/admin)"
    echo ""
    echo "Next: Start traffic with './scripts/start_traffic.sh normal'"
fi

exit $failed

