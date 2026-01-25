#!/bin/bash
# Logs Viewer - Follow logs from services
# Usage: ./scripts/logs.sh [service-name] [--level LEVEL]
# Examples:
#   ./scripts/logs.sh                    # All services
#   ./scripts/logs.sh user-service       # Specific service
#   ./scripts/logs.sh --level ERROR      # Filter by log level

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SERVICE=""
LOG_LEVEL=""
FOLLOW=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --no-follow)
            FOLLOW=false
            shift
            ;;
        -*)
            echo -e "${RED}[ERROR]${NC} Unknown option: $1"
            echo "Usage: ./scripts/logs.sh [service-name] [--level LEVEL] [--no-follow]"
            exit 1
            ;;
        *)
            SERVICE="$1"
            shift
            ;;
    esac
done

echo "================================================================================"
echo "  LOGS VIEWER"
echo "================================================================================"
echo ""

# List available services if no service specified
if [ -z "$SERVICE" ]; then
    echo -e "${BLUE}[INFO]${NC} Showing logs from all services"
    echo ""

    if [ "$FOLLOW" = true ]; then
        docker-compose logs -f --tail=50
    else
        docker-compose logs --tail=50
    fi
else
    # Check if service exists
    if ! docker-compose ps --services | grep -q "^${SERVICE}$"; then
        echo -e "${RED}[ERROR]${NC} Service not found: ${SERVICE}"
        echo ""
        echo "Available services:"
        docker-compose ps --services | sed 's/^/  - /'
        exit 1
    fi

    echo -e "${BLUE}[INFO]${NC} Showing logs from: ${SERVICE}"
    echo ""

    if [ "$FOLLOW" = true ]; then
        if [ -n "$LOG_LEVEL" ]; then
            docker-compose logs -f --tail=50 "${SERVICE}" | grep -i "${LOG_LEVEL}"
        else
            docker-compose logs -f --tail=50 "${SERVICE}"
        fi
    else
        if [ -n "$LOG_LEVEL" ]; then
            docker-compose logs --tail=50 "${SERVICE}" | grep -i "${LOG_LEVEL}"
        else
            docker-compose logs --tail=50 "${SERVICE}"
        fi
    fi
fi

