#!/bin/bash
# Start Traffic Generator - Generate realistic user traffic
# Usage: ./scripts/start_traffic.sh [profile]
# Profiles: normal, peak, degraded, burst

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Default profile
PROFILE=${1:-normal}

# Valid profiles
VALID_PROFILES=("normal" "peak" "degraded" "burst")

# Check if profile is valid
if [[ ! " ${VALID_PROFILES[@]} " =~ " ${PROFILE} " ]]; then
    echo -e "${RED}[ERROR]${NC} Invalid profile: ${PROFILE}"
    echo "Valid profiles: ${VALID_PROFILES[*]}"
    exit 1
fi

echo "================================================================================"
echo "  TRAFFIC GENERATOR - Starting with profile: ${PROFILE}"
echo "================================================================================"
echo ""

# Check if services are running
echo -e "${BLUE}[INFO]${NC} Checking if services are running..."
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${RED}[ERROR]${NC} Services are not running. Please run './scripts/deploy_simulator.sh' first."
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Services are running"

# Stop any existing traffic generator
echo ""
echo -e "${BLUE}[INFO]${NC} Stopping any existing traffic generator..."
docker-compose --profile traffic stop traffic-generator > /dev/null 2>&1 || true
docker-compose --profile traffic rm -f traffic-generator > /dev/null 2>&1 || true

# Start traffic generator with profile
echo ""
echo -e "${YELLOW}[STARTING]${NC} Traffic generator with profile: ${PROFILE}"
export TRAFFIC_PROFILE=${PROFILE}
docker-compose --profile traffic up -d traffic-generator

echo ""
echo -e "${BLUE}[INFO]${NC} Traffic generator started. Monitoring logs..."
echo ""
echo "================================================================================"
echo ""

# Follow logs
docker-compose --profile traffic logs -f traffic-generator

# Note: This will run until interrupted (Ctrl+C)

