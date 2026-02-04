# Docker Deployment Guide

Complete Docker setup for the Markov RL API Cache simulator environment.

## Overview

This deployment includes:
- **7 Microservices** - Auth, User, Product, Cart, Order, Payment, Inventory
- **Infrastructure** - Redis cache, Prometheus, Grafana
- **Traffic Generator** - Realistic user behavior simulation
- **Monitoring** - Real-time dashboards and metrics

## Quick Start

### Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Docker Compose v2.0+
- 8GB+ RAM available for Docker
- Ports 3000, 6379, 8001-8007, 9090 available

### Deploy Everything

**Windows (PowerShell):**
```powershell
.\scripts\deploy_simulator.ps1
```

**Linux/Mac (Bash):**
```bash
chmod +x scripts/*.sh
./scripts/deploy_simulator.sh
```

This will:
1. Build Docker images
2. Start infrastructure (Redis, Prometheus, Grafana)
3. Wait for health checks
4. Start all microservices
5. Display service URLs

## Services

### Microservices

| Service | Port | Description |
|---------|------|-------------|
| auth-service | 8002 | Authentication & tokens |
| user-service | 8001 | User management |
| product-service | 8003 | Product catalog |
| cart-service | 8004 | Shopping cart |
| order-service | 8005 | Order processing |
| payment-service | 8006 | Payment handling |
| inventory-service | 8007 | Stock management |

### Infrastructure

| Service | Port | Credentials |
|---------|------|-------------|
| Redis | 6379 | - |
| Prometheus | 9090 | - |
| Grafana | 3000 | admin/admin |

## Usage

### Check Status

```bash
docker-compose ps
```

### Verify Services

```bash
# Health checks
curl http://localhost:8001/health  # User service
curl http://localhost:8002/health  # Auth service
curl http://localhost:8003/health  # Product service

# Prometheus targets
curl http://localhost:9090/targets
```

### Start Traffic Generation

**Windows:**
```powershell
.\scripts\start_traffic.ps1 normal   # Normal traffic
.\scripts\start_traffic.ps1 peak     # Peak hours
.\scripts\start_traffic.ps1 burst    # Burst traffic
.\scripts\start_traffic.ps1 degraded # Degraded performance
```

**Linux/Mac:**
```bash
./scripts/start_traffic.sh normal
./scripts/start_traffic.sh peak
```

### View Logs

**All services:**
```powershell
.\scripts\logs.ps1
```

**Specific service:**
```powershell
.\scripts\logs.ps1 -Service user-service
```

**Filter by level:**
```powershell
.\scripts\logs.ps1 -Service order-service -Level ERROR
```

### Stop Everything

**Without removing data:**
```powershell
.\scripts\stop_all.ps1
```

**Clean shutdown (removes volumes):**
```powershell
.\scripts\stop_all.ps1 -Clean
```

## Monitoring

### Grafana Dashboards

Access Grafana at http://localhost:3000 (admin/admin)

**Pre-configured dashboards:**
- System Overview - Request rates, latency, errors
- Cache Performance - Hit rates, evictions, prefetch
- RL Agent - Actions, rewards, Q-values

### Prometheus Metrics

Access Prometheus at http://localhost:9090

**Available metrics:**
- Request rates per service
- Latency percentiles (p50, p95, p99)
- Error rates
- Cache statistics
- Resource utilization

## Development

### Hot Reload

The `docker-compose.override.yml` enables hot reload in development:
- Source code mounted as volumes
- Debug mode enabled
- Relaxed resource limits

Edit code locally and services automatically reload.

### Custom Configuration

**Override environment variables:**
```bash
export REDIS_URL=redis://custom-redis:6379
docker-compose up -d
```

**Use custom compose file:**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Architecture

### Network

All services communicate on the `simulator-network` bridge network:
- Subnet: 172.20.0.0/16
- Internal DNS resolution
- Isolated from host network

### Volumes

Persistent data stored in Docker volumes:
- `redis-data` - Cache data
- `prometheus-data` - Metrics history (7 days)
- `grafana-data` - Dashboards and config

### Health Checks

All services have health checks:
- Interval: 15s
- Timeout: 5s
- Start period: 30s
- Retries: 3

Services start only after dependencies are healthy.

### Resource Limits

**Production limits:**
- Auth/Inventory: 256MB RAM, 0.25 CPU
- Services: 512MB RAM, 0.5 CPU
- Redis: 512MB RAM, 0.5 CPU
- Prometheus: 1GB RAM, 1.0 CPU
- Grafana: 512MB RAM, 0.5 CPU

**Development (override):**
- All services: 1-2GB RAM, 2.0 CPU

## Troubleshooting

### Services Won't Start

**Check Docker resources:**
```bash
docker info
```

Ensure at least 8GB RAM allocated to Docker.

**Check logs:**
```powershell
.\scripts\logs.ps1 -Service <failing-service>
```

**Rebuild images:**
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Port Conflicts

If ports are in use:
1. Stop conflicting services
2. Or edit `docker-compose.yml` to use different ports

### Services Unhealthy

**Check health:**
```bash
docker-compose ps
```

**Restart specific service:**
```bash
docker-compose restart user-service
```

**Full restart:**
```bash
.\scripts\stop_all.ps1
.\scripts\deploy_simulator.ps1
```

### Out of Memory

**Increase Docker memory:**
- Docker Desktop → Settings → Resources → Memory
- Allocate at least 8GB

**Or reduce resource usage:**
- Edit `docker-compose.yml`
- Lower memory limits
- Stop unused services

### Clean Slate

**Remove everything:**
```bash
docker-compose down -v --remove-orphans
docker system prune -a --volumes
.\scripts\deploy_simulator.ps1
```

## Advanced

### Custom Traffic Profiles

Create custom profile in `simulator/traffic/profiles/my-profile.yaml`:
```yaml
name: "my-profile"
requests_per_second: 500
duration_seconds: 600
user_type_distribution:
  premium: 0.3
  free: 0.6
  guest: 0.1
workflow_distribution:
  browse: 0.4
  purchase: 0.3
  account: 0.2
  quickbuy: 0.1
```

Run with:
```bash
export TRAFFIC_PROFILE=my-profile
docker-compose --profile traffic up traffic-generator
```

### Failure Injection

Use the failure injection system to test resilience:

```python
# From inside a container
from simulator.failures.injector import FailureInjector
injector = FailureInjector(services)
injector.inject_latency_spike('payment-service', multiplier=5.0)
```

### Production Deployment

For production:
1. Remove `docker-compose.override.yml`
2. Use secrets for credentials
3. Enable TLS/SSL
4. Configure reverse proxy (nginx)
5. Set up external monitoring
6. Use orchestration (Kubernetes)

## Scripts Reference

### Windows (PowerShell)

- `deploy_simulator.ps1` - Deploy everything
- `start_traffic.ps1 [profile]` - Start traffic generator
- `stop_all.ps1 [-Clean]` - Stop services
- `logs.ps1 [-Service] [-Level]` - View logs

### Linux/Mac (Bash)

- `deploy_simulator.sh` - Deploy everything
- `start_traffic.sh [profile]` - Start traffic generator
- `stop_all.sh [--clean]` - Stop services
- `logs.sh [service] [--level LEVEL]` - View logs

## Files Structure

```
.
├── Dockerfile                      # Multi-stage build
├── docker-compose.yml              # Main configuration
├── docker-compose.override.yml     # Development overrides
├── .dockerignore                   # Exclude from build
├── requirements.txt                # Python dependencies
├── monitoring/
│   ├── prometheus.yml              # Metrics scraping
│   ├── datasources/                # Grafana datasources
│   └── dashboards/                 # Grafana dashboards
└── scripts/
    ├── deploy_simulator.{sh,ps1}   # Deployment
    ├── start_traffic.{sh,ps1}      # Traffic generation
    ├── stop_all.{sh,ps1}           # Shutdown
    └── logs.{sh,ps1}               # Log viewing
```

## Support

For issues or questions:
1. Check logs: `.\scripts\logs.ps1 -Service <service>`
2. Verify health: `docker-compose ps`
3. Check resources: `docker stats`
4. Review documentation in this README

## License

See LICENSE file in project root.

