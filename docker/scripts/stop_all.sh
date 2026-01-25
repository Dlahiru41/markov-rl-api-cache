#!/bin/bash
# Stop All Services - Gracefully stop all containers
# Usage: ./scripts/stop_all.sh [--clean]
# Options:
#   --clean    Also remove volumes (WARNING: deletes all data)

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            echo "Usage: ./scripts/stop_all.sh [--clean]"
            exit 1
            ;;
    esac
done

echo "================================================================================"
echo "  STOPPING ALL SERVICES"
echo "================================================================================"
echo ""

# Stop traffic generator first (if running)
echo -e "${BLUE}[INFO]${NC} Stopping traffic generator..."
docker-compose --profile traffic stop traffic-generator > /dev/null 2>&1 || true
docker-compose --profile traffic rm -f traffic-generator > /dev/null 2>&1 || true
echo -e "${GREEN}[OK]${NC} Traffic generator stopped"

# Stop application services
echo ""
echo -e "${BLUE}[INFO]${NC} Stopping application services..."
docker-compose stop \
    order-service \
    payment-service \
    cart-service \
    product-service \
    inventory-service \
    user-service \
    auth-service
echo -e "${GREEN}[OK]${NC} Application services stopped"

# Stop infrastructure services
echo ""
echo -e "${BLUE}[INFO]${NC} Stopping infrastructure services..."
docker-compose stop grafana prometheus redis
echo -e "${GREEN}[OK]${NC} Infrastructure services stopped"

# Remove containers
echo ""
echo -e "${BLUE}[INFO]${NC} Removing containers..."
docker-compose down
echo -e "${GREEN}[OK]${NC} Containers removed"

# Clean volumes if requested
if [ "$CLEAN" = true ]; then
    echo ""
    echo -e "${YELLOW}[WARNING]${NC} Removing volumes (this will delete all data)..."
    read -p "Are you sure? (yes/no): " -r
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        docker-compose down -v
        echo -e "${GREEN}[OK]${NC} Volumes removed"
    else
        echo -e "${BLUE}[INFO]${NC} Volume removal cancelled"
    fi
fi

# Show remaining resources
echo ""
echo -e "${BLUE}[INFO]${NC} Checking for remaining resources..."
CONTAINERS=$(docker ps -a --filter "name=simulator-" --format "{{.Names}}" | wc -l)
VOLUMES=$(docker volume ls --filter "name=markov-rl-api-cache" --format "{{.Name}}" | wc -l)

if [ $CONTAINERS -gt 0 ]; then
    echo -e "${YELLOW}[WARNING]${NC} Some containers still exist:"
    docker ps -a --filter "name=simulator-" --format "  - {{.Names}} ({{.Status}})"
fi

if [ $VOLUMES -gt 0 ]; then
    echo -e "${BLUE}[INFO]${NC} Persistent volumes still exist:"
    docker volume ls --filter "name=markov-rl-api-cache" --format "  - {{.Name}}"
    echo ""
    echo "To remove volumes: ./scripts/stop_all.sh --clean"
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}[SUCCESS]${NC} All services stopped"
echo "================================================================================"
echo ""
echo "To start again: ./scripts/deploy_simulator.sh"
echo ""

