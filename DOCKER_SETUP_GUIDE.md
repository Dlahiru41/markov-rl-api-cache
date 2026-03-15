# Docker-Only Deployment Guide: Markov RL API Cache
## Local Development & Testing Setup

**Version**: 1.0  
**Date**: March 15, 2026  
**Scope**: Docker-based local setup ONLY  
**Target Audience**: Developers, DevOps Engineers

---

## Quick Start (5 Minutes)

```bash
# Clone and navigate to project
cd markov-rl-api-cache

# Start all services with Docker Compose
docker-compose -f docker/docker-compose.yml up -d

# Verify services are running
docker-compose ps

# Check health
curl http://localhost:8000/health
# Response: {"status": "healthy"}
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Environment Setup](#docker-environment-setup)
3. [Starting Services](#starting-services)
4. [Accessing Services](#accessing-services)
5. [Training Models](#training-models)
6. [Testing & Validation](#testing--validation)
7. [Integration](#integration)
8. [Monitoring](#monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Cleanup](#cleanup)

---

## Prerequisites

### Required Software

```
✅ Docker Desktop (or Docker Engine)
   └─ Windows: Docker Desktop for Windows
   └─ Mac: Docker Desktop for Mac
   └─ Linux: Docker Engine + Docker Compose

✅ Python 3.10+ (for local model training - optional)
   └─ Install: python --version should show 3.10+

✅ Git (to clone repository)
   └─ Install: git --version

✅ 8+ GB RAM available
   └─ Docker will need 4-6 GB
```

### System Requirements

```
Minimum:
├─ CPU: 2 cores
├─ RAM: 8 GB
├─ Disk: 20 GB SSD
└─ OS: Windows 10+, macOS 10.15+, Ubuntu 18.04+

Recommended:
├─ CPU: 4+ cores
├─ RAM: 16 GB
├─ Disk: 50 GB SSD
└─ OS: Latest version
```

### Installation Check

```bash
# Verify Docker installation
docker --version
# Expected: Docker version 20.10+

# Verify Docker Compose
docker-compose --version
# Expected: Docker Compose version 1.29+

# Verify Python (optional)
python --version
# Expected: Python 3.10+
```

---

## Docker Environment Setup

### 1. Clone Repository

```bash
# Clone the project
git clone https://github.com/yourcompany/markov-rl-api-cache.git
cd markov-rl-api-cache

# Verify you have docker files
ls docker/
# Should show: Dockerfile, docker-compose.yml, .dockerignore
```

### 2. Create Environment File

```bash
# Create .env file from example
cp .env.example .env

# Edit .env if needed (optional - defaults work fine)
# nano .env  (on Linux/Mac)
# notepad .env  (on Windows)
```

**Default .env values** (fine for local development):
```dotenv
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

API_HOST=0.0.0.0
API_PORT=8000

PROMETHEUS_PORT=8001
ENABLE_MONITORING=true
LOG_LEVEL=INFO
```

### 3. Build Docker Image (First Time Only)

```bash
# Build the Docker image
docker-compose -f docker/docker-compose.yml build

# This will take 2-5 minutes
# Downloads Python 3.10, installs dependencies
```

---

## Starting Services

### Option 1: Start All Services (Recommended)

```bash
# Navigate to docker directory
cd docker

# Start all services in background
docker-compose up -d

# Expected output:
# Creating network "docker_simulator-network" with driver "bridge"
# Creating simulator-redis ... done
# Creating simulator-prometheus ... done
# Creating simulator-grafana ... done
# Creating simulator-rl-cache ... done
```

### Option 2: Start Specific Services

```bash
# Start only Redis
docker-compose up -d redis

# Start Redis + Cache Intelligence
docker-compose up -d redis cache-intelligence

# Start everything except Grafana
docker-compose up -d redis prometheus cache-intelligence
```

### Option 3: Run in Foreground (for debugging)

```bash
# Start all services and show logs
docker-compose up

# Press Ctrl+C to stop all services
```

### Verify Services are Running

```bash
# Check status
docker-compose ps

# Expected output:
# NAME                STATUS
# ────────────────────────────────────────
# simulator-redis     Up 2 minutes
# simulator-prometheus Up 2 minutes
# simulator-grafana   Up 2 minutes
# simulator-rl-cache  Up 2 minutes
```

---

## Accessing Services

### 1. Cache Intelligence API

```
URL: http://localhost:8000
Status endpoint: http://localhost:8000/health
API docs: http://localhost:8000/docs
Redoc docs: http://localhost:8000/redoc
```

**Test it:**
```bash
curl http://localhost:8000/health
# Response: {"status": "healthy", "timestamp": "2026-03-15T10:30:00"}
```

### 2. Redis Cache

```
Host: localhost
Port: 6379
Database: 0
CLI access: redis-cli
```

**Connect via CLI:**
```bash
# Using docker exec
docker exec -it simulator-redis redis-cli

# Commands:
redis> PING
# Response: PONG

redis> DBSIZE
# Response: (integer) 0

redis> exit
```

### 3. Prometheus Metrics

```
URL: http://localhost:9090
Query interface for metrics
Scrapes data every 15 seconds
```

**Example queries:**
```
markov_rl_cache_hits_total
markov_rl_cache_misses_total
markov_rl_api_request_duration_ms
```

### 4. Grafana Dashboards

```
URL: http://localhost:3000
Username: admin
Password: admin
```

**First time setup:**
1. Open http://localhost:3000
2. Login with admin/admin
3. Change password (optional)
4. Dashboards are auto-provisioned
5. View "Markov RL Cache Intelligence" dashboard

---

## Training Models

### Step 1: Prepare Training Data

```bash
# Create data directory if it doesn't exist
mkdir -p data/training

# Add your API sequences to data/training/sequences.json
# Format:
# [
#   {
#     "user_type": "premium",
#     "sequence": ["GET /api/products", "GET /api/products/123", ...],
#     "timestamp": "2026-03-15T10:30:00"
#   },
#   ...
# ]
```

### Step 2: Train Markov Model

```bash
# Using Docker (recommended)
docker exec -it simulator-rl-cache python src/markov/train.py \
  --data data/training/sequences.json \
  --output models/markov.pkl

# Or locally (if Python 3.10+ installed)
python src/markov/train.py \
  --data data/training/sequences.json \
  --output models/markov.pkl

# Expected output:
# Training Markov predictor...
# Processed 5000 sequences
# Vocabulary size: 42
# Top-1 accuracy: 68.5%
# Model saved: models/markov.pkl
```

### Step 3: Train DQN Agent

```bash
# Using Docker
docker exec -it simulator-rl-cache python train_rl_agents.py \
  --episodes 500 \
  --output models/dqn_agent.pt

# Expected output:
# Starting training...
# Episode 50: reward=245.3, epsilon=0.90
# Episode 100: reward=287.4, epsilon=0.81
# Episode 200: reward=342.1, epsilon=0.65
# Episode 500: reward=452.3, epsilon=0.15
# Model saved: models/dqn_agent.pt
```

### Step 4: Validate Models

```bash
# Run validation tests
docker exec -it simulator-rl-cache python test_agent.pt

# Expected output:
# ✓ Model loads successfully
# ✓ Predictions working
# ✓ Cache operations working
# ✓ All tests passed
```

---

## Testing & Validation

### Test 1: Cache Operations

```bash
# Test cache set/get
curl -X POST http://localhost:8000/cache/set \
  -H "Content-Type: application/json" \
  -d '{
    "key": "test_key",
    "data": {"value": 42},
    "ttl_seconds": 3600
  }'

# Verify it was stored
curl http://localhost:8000/cache/get?key=test_key
# Response: {"hit": true, "data": {"value": 42}, "ttl_remaining_seconds": 3599}

# Delete the key
curl -X DELETE http://localhost:8000/cache/delete?key=test_key
```

### Test 2: Model Predictions

```bash
# Get cache decision
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint": "GET /api/products",
    "user_type": "premium"
  }'

# Response:
# {
#   "action": "serve_from_cache",
#   "ttl_seconds": 3600,
#   "confidence": 0.82
# }
```

### Test 3: Load Test

```bash
# Using Apache Bench (if installed)
ab -n 1000 -c 10 http://localhost:8000/health

# Expected output:
# Requests per second: 500+
# Mean time per request: 20ms

# Or using curl loop
for i in {1..100}; do
  curl -s http://localhost:8000/health > /dev/null
  echo "Request $i completed"
done
```

### Test 4: Full Integration Test

```bash
# Run the complete test suite
docker exec -it simulator-rl-cache pytest tests/ -v

# Expected output:
# test_cache_operations.py::test_set_get PASSED
# test_markov_predictor.py::test_prediction PASSED
# test_dqn_agent.py::test_action_selection PASSED
# ===================== 25 passed in 3.45s ======================
```

---

## Integration

### How to Integrate with Your Application

**Option A: Using the REST API**

```python
# Python example
import requests

cache_service = "http://localhost:8000"

# Get cache decision
response = requests.post(
    f"{cache_service}/decide",
    json={
        "endpoint": "GET /api/products/123",
        "user_type": "premium"
    }
)
action = response.json()

if action['action'] == 'serve_from_cache':
    # Serve from cache
    cached = requests.get(
        f"{cache_service}/cache/get",
        params={"key": "api:products:123"}
    )
    return cached.json()['data']
else:
    # Fetch from backend and cache
    data = fetch_from_backend()
    requests.post(
        f"{cache_service}/cache/set",
        json={"key": "api:products:123", "data": data, "ttl_seconds": 3600}
    )
    return data
```

**Option B: Using Docker Network**

```yaml
# In your docker-compose.yml
services:
  my-app:
    image: my-app:latest
    depends_on:
      - cache-intelligence
    networks:
      - simulator-network
    environment:
      CACHE_SERVICE_URL: http://cache-intelligence:8000

networks:
  simulator-network:
    external: true
    # Created by markov-rl-api-cache docker-compose
```

---

## Monitoring

### View Real-Time Metrics

**Grafana Dashboard:**
1. Open http://localhost:3000
2. Login: admin/admin
3. Navigate to "Markov RL Cache Intelligence" dashboard
4. View metrics in real-time:
   - Cache hit rate
   - API latency
   - System health
   - Model performance

**Prometheus Query Examples:**

```promql
# Cache hit rate (%)
rate(markov_rl_cache_hits_total[5m]) / 
(rate(markov_rl_cache_hits_total[5m]) + 
 rate(markov_rl_cache_misses_total[5m])) * 100

# Average latency (ms)
rate(markov_rl_api_request_duration_ms_sum[5m]) /
rate(markov_rl_api_request_duration_ms_count[5m])

# Cache size
markov_rl_cache_size_bytes

# Error rate
rate(markov_rl_api_errors_total[5m])
```

### View Logs

```bash
# View cache intelligence logs
docker-compose logs cache-intelligence

# Follow logs in real-time
docker-compose logs -f cache-intelligence

# View last 100 lines
docker-compose logs cache-intelligence --tail 100

# View specific service
docker-compose logs redis
docker-compose logs prometheus
docker-compose logs grafana
```

---

## Troubleshooting

### Problem: Services won't start

```bash
# Check if ports are in use
# macOS/Linux:
lsof -i :8000
lsof -i :6379
lsof -i :9090
lsof -i :3000

# Windows:
netstat -ano | findstr :8000

# Solution: Change ports in docker-compose.yml
# Or kill the process using the port
```

### Problem: Redis connection failed

```bash
# Check Redis status
docker-compose ps redis

# Connect to Redis
docker exec -it simulator-redis redis-cli ping
# Should return: PONG

# View Redis logs
docker-compose logs redis

# Restart Redis
docker-compose restart redis
```

### Problem: Cache Intelligence API not responding

```bash
# Check service status
docker-compose ps cache-intelligence

# View logs
docker-compose logs cache-intelligence

# Check health endpoint
curl http://localhost:8000/health

# Restart service
docker-compose restart cache-intelligence
```

### Problem: Out of memory

```bash
# Check Docker resource limits
docker stats

# If services using too much memory:
# 1. Increase Docker memory limit (Docker Desktop Settings)
# 2. Or reduce cache size in environment
# 3. Or stop unused services (Grafana uses 500MB+)
```

### Problem: Models not training

```bash
# Check if data file exists
ls -la data/training/sequences.json

# Run with verbose output
docker exec -it simulator-rl-cache python train_rl_agents.py \
  --episodes 100 \
  --verbose

# Check GPU availability (optional)
docker exec -it simulator-rl-cache python -c "import torch; print(torch.cuda.is_available())"
```

---

## Cleanup

### Stop All Services

```bash
# Stop all services (keep volumes)
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Remove images too
docker-compose down -v --rmi local
```

### Free Up Space

```bash
# Remove unused Docker containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Full cleanup (careful!)
docker system prune -a --volumes
```

### Restart Everything Fresh

```bash
# Complete reset
docker-compose down -v
docker volume prune -f
docker-compose up -d

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

---

## Docker Compose Configuration Reference

**Key services in docker-compose.yml:**

```yaml
services:
  # Cache Intelligence API
  cache-intelligence:
    image: markov-rl-cache:latest
    ports: ["8000:8000"]  # API endpoint
    depends_on: [redis]
    environment:
      REDIS_HOST: redis
      API_PORT: 8000
  
  # Redis Cache Backend
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: redis-server --maxmemory 512mb
  
  # Prometheus Metrics
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes: ["./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml"]
  
  # Grafana Dashboards
  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
```

---

## Summary

✅ Docker-based local setup  
✅ All services run locally  
✅ No production complexity  
✅ Easy start/stop/restart  
✅ Integrated monitoring  
✅ Ready for testing & development  

**Next steps:**
1. Follow "Quick Start" section above
2. Train your models
3. Run integration tests
4. Monitor metrics in Grafana
5. Integrate with your application

---

**For questions or issues, refer to the Troubleshooting section above.**


