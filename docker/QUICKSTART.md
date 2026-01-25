# Docker Deployment

All Docker-related files and scripts are organized in this directory.

## Directory Structure

```
docker/
├── Dockerfile                      # Multi-stage build configuration
├── docker-compose.yml              # Main orchestration file
├── docker-compose.override.yml     # Development overrides
├── .dockerignore                   # Build exclusions
├── README.md                       # This file
├── DOCKER_README.md               # Complete Docker documentation
│
├── scripts/                        # Deployment scripts
│   ├── deploy_simulator.sh         # Deploy all services (Bash)
│   ├── deploy_simulator.ps1        # Deploy all services (PowerShell)
│   ├── start_traffic.sh            # Start traffic generator (Bash)
│   ├── start_traffic.ps1           # Start traffic generator (PowerShell)
│   ├── stop_all.sh                 # Stop all services (Bash)
│   ├── stop_all.ps1                # Stop all services (PowerShell)
│   ├── logs.sh                     # View logs (Bash)
│   ├── logs.ps1                    # View logs (PowerShell)
│   ├── validate_deployment.sh      # Validation (Bash)
│   └── validate_deployment.ps1     # Validation (PowerShell)
│
└── monitoring/                     # Monitoring configuration
    ├── prometheus.yml              # Prometheus scrape config
    └── datasources/                # Grafana datasources
        └── prometheus.yml          # Prometheus datasource

```

## Quick Start

### Windows (PowerShell)

From the project root:
```powershell
cd docker
.\scripts\deploy_simulator.ps1
```

### Linux/Mac (Bash)

From the project root:
```bash
cd docker
chmod +x scripts/*.sh
./scripts/deploy_simulator.sh
```

## Commands

All commands should be run from the `docker/` directory.

### Deploy Everything
```bash
# Windows
.\scripts\deploy_simulator.ps1

# Linux/Mac
./scripts/deploy_simulator.sh
```

### Start Traffic Generation
```bash
# Windows
.\scripts\start_traffic.ps1 normal

# Linux/Mac
./scripts/start_traffic.sh normal
```

Profiles: `normal`, `peak`, `degraded`, `burst`

### View Logs
```bash
# Windows
.\scripts\logs.ps1 -Service user-service

# Linux/Mac
./scripts/logs.sh user-service
```

### Stop All Services
```bash
# Windows
.\scripts\stop_all.ps1

# Linux/Mac
./scripts/stop_all.sh
```

Add `-Clean` (PowerShell) or `--clean` (Bash) to remove volumes.

### Validate Deployment
```bash
# Windows
.\scripts\validate_deployment.ps1

# Linux/Mac
./scripts/validate_deployment.sh
```

## Services

| Service | Port | URL |
|---------|------|-----|
| Auth Service | 8002 | http://localhost:8002 |
| User Service | 8001 | http://localhost:8001 |
| Product Service | 8003 | http://localhost:8003 |
| Cart Service | 8004 | http://localhost:8004 |
| Order Service | 8005 | http://localhost:8005 |
| Payment Service | 8006 | http://localhost:8006 |
| Inventory Service | 8007 | http://localhost:8007 |
| Redis | 6379 | redis://localhost:6379 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |

**Grafana Credentials:** admin/admin

## Architecture

### Network
- All services run on `simulator-network` (172.20.0.0/16)
- Internal DNS resolution between services
- External access via published ports

### Volumes
- `redis-data` - Redis persistence
- `prometheus-data` - Metrics storage (7 days)
- `grafana-data` - Dashboards and configuration

### Health Checks
- All services have health checks
- 15-second intervals
- Automatic restart on failure
- Startup dependencies enforced

## Development

### Hot Reload

Edit code in the parent `simulator/` directory and changes will automatically reload (no rebuild needed).

The `docker-compose.override.yml` file enables:
- Source code volume mounts
- Debug mode
- Relaxed resource limits

### Debug Mode

Services run with `DEBUG=true` and detailed logging enabled.

### Resource Limits

**Production (docker-compose.yml):**
- Services: 256-512MB RAM
- Infrastructure: 512MB-1GB RAM

**Development (docker-compose.override.yml):**
- All services: 1-2GB RAM
- More generous CPU allocation

## Troubleshooting

### Services Won't Start

1. Check Docker is running: `docker info`
2. Check logs: `.\scripts\logs.ps1 -Service <service-name>`
3. Rebuild: `docker-compose build --no-cache`

### Port Conflicts

If ports are already in use, edit `docker-compose.yml` to change port mappings.

### Out of Memory

Increase Docker memory allocation:
- Docker Desktop → Settings → Resources → Memory
- Allocate at least 8GB

### Clean Slate

Remove everything and start fresh:
```bash
# Windows
.\scripts\stop_all.ps1 -Clean
docker system prune -a --volumes -f
.\scripts\deploy_simulator.ps1

# Linux/Mac
./scripts/stop_all.sh --clean
docker system prune -a --volumes -f
./scripts/deploy_simulator.sh
```

## Documentation

See `README.md` in this directory for complete documentation including:
- Detailed architecture
- Advanced usage
- Monitoring setup
- Production deployment
- Troubleshooting guide

## Notes

- All scripts work from the `docker/` directory
- Build context is set to parent directory (`..`)
- Monitoring configs are in `monitoring/` subdirectory
- Scripts are available for both Windows (PowerShell) and Linux/Mac (Bash)

