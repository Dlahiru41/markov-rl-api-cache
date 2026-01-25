#!/bin/bash
# Deploy Simulator - Build and start all services
# Usage: ./scripts/deploy_simulator.sh

set -e  # Exit on error

echo "================================================================================"
echo "  MARKOV RL API CACHE - SIMULATOR DEPLOYMENT"
echo "================================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}[ERROR]${NC} Docker is not running. Please start Docker first."
    exit 1
fi

echo -e "${BLUE}[INFO]${NC} Docker is running"

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}[ERROR]${NC} docker-compose not found. Please install docker-compose first."
    exit 1
fi

echo -e "${BLUE}[INFO]${NC} docker-compose found"

# Build images if needed
echo ""
echo -e "${YELLOW}[STEP 1]${NC} Building Docker images..."
docker-compose build --pull

echo ""
echo -e "${YELLOW}[STEP 2]${NC} Starting infrastructure services (Redis, Prometheus, Grafana)..."
docker-compose up -d redis prometheus grafana

echo ""
echo -e "${BLUE}[INFO]${NC} Waiting for infrastructure to be healthy..."
sleep 5

# Wait for Redis
echo -n "  - Waiting for Redis... "
timeout=60
elapsed=0
while ! docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ $elapsed -ge $timeout ]; then
        echo -e "${RED}TIMEOUT${NC}"
        echo -e "${RED}[ERROR]${NC} Redis failed to start"
        exit 1
    fi
done
echo -e "${GREEN}OK${NC}"

# Wait for Prometheus
echo -n "  - Waiting for Prometheus... "
elapsed=0
while ! curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ $elapsed -ge $timeout ]; then
        echo -e "${RED}TIMEOUT${NC}"
        echo -e "${RED}[ERROR]${NC} Prometheus failed to start"
        exit 1
    fi
done
echo -e "${GREEN}OK${NC}"

# Wait for Grafana
echo -n "  - Waiting for Grafana... "
elapsed=0
while ! curl -sf http://localhost:3000/api/health > /dev/null 2>&1; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ $elapsed -ge $timeout ]; then
        echo -e "${RED}TIMEOUT${NC}"
        echo -e "${RED}[ERROR]${NC} Grafana failed to start"
        exit 1
    fi
done
echo -e "${GREEN}OK${NC}"

echo ""
echo -e "${YELLOW}[STEP 3]${NC} Starting microservices..."
docker-compose up -d \
    auth-service \
    user-service \
    inventory-service \
    product-service \
    cart-service \
    payment-service \
    order-service

echo ""
echo -e "${BLUE}[INFO]${NC} Waiting for all services to be healthy..."
sleep 10

# Function to check service health
check_service() {
    local service=$1
    local port=$2
    local max_attempts=30
    local attempt=0

    echo -n "  - Checking ${service}... "

    while [ $attempt -lt $max_attempts ]; do
        if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
            echo -e "${GREEN}OK${NC}"
            return 0
        fi
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}TIMEOUT${NC}"
    return 1
}

# Check all services
failed_services=()

check_service "auth-service" 8002 || failed_services+=("auth-service")
check_service "user-service" 8001 || failed_services+=("user-service")
check_service "inventory-service" 8007 || failed_services+=("inventory-service")
check_service "product-service" 8003 || failed_services+=("product-service")
check_service "cart-service" 8004 || failed_services+=("cart-service")
check_service "payment-service" 8006 || failed_services+=("payment-service")
check_service "order-service" 8005 || failed_services+=("order-service")

echo ""
if [ ${#failed_services[@]} -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS]${NC} All services are healthy!"
else
    echo -e "${RED}[WARNING]${NC} Some services failed to start:"
    for service in "${failed_services[@]}"; do
        echo -e "  - ${RED}$service${NC}"
    done
    echo ""
    echo "Run './scripts/logs.sh' to see logs"
fi

echo ""
echo "================================================================================"
echo "  DEPLOYMENT COMPLETE"
echo "================================================================================"
echo ""
echo -e "${GREEN}Services Available:${NC}"
echo "  - Auth Service:      http://localhost:8002"
echo "  - User Service:      http://localhost:8001"
echo "  - Product Service:   http://localhost:8003"
echo "  - Cart Service:      http://localhost:8004"
echo "  - Order Service:     http://localhost:8005"
echo "  - Payment Service:   http://localhost:8006"
echo "  - Inventory Service: http://localhost:8007"
echo ""
echo -e "${GREEN}Monitoring:${NC}"
echo "  - Prometheus:        http://localhost:9090"
echo "  - Grafana:           http://localhost:3000 (admin/admin)"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Check status:     docker-compose ps"
echo "  2. Start traffic:    ./scripts/start_traffic.sh normal"
echo "  3. View logs:        ./scripts/logs.sh [service-name]"
echo "  4. Stop all:         ./scripts/stop_all.sh"
echo ""
echo "================================================================================"

