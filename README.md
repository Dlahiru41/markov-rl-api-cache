# markov-rl-api-cache

Markov Chain-based Reinforcement Learning framework for adaptive API caching in microservices.

This service acts as a **transparent API Gateway / Reverse Proxy**. Point it at
any upstream microservice and all traffic flows through it — GET responses are
cached in Redis and the built-in Markov + RL intelligence prefetches predicted
next requests automatically. **Zero code changes** are required on the upstream
or the client.

```
Client ─► markov-rl-cache:8000/api/users/123
               │  cache MISS
               ▼
          upstream:9000/api/users/123
               │  store in Redis
               ▼
          return to client (X-Cache: MISS)

Client ─► markov-rl-cache:8000/api/users/123  (next call)
               │  cache HIT
               ▼
          return from Redis (X-Cache: HIT, upstream never called)
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start — Local Development](#quick-start--local-development)
3. [Quick Start — Docker Compose](#quick-start--docker-compose)
4. [Configuration](#configuration)
5. [Verifying the Service](#verifying-the-service)
6. [Admin Endpoints](#admin-endpoints)
7. [Running Tests](#running-tests)
8. [Architecture Overview](#architecture-overview)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Dependency | Version | Required for |
|------------|---------|-------------|
| **Python** | 3.10+ | Local development |
| **Redis** | 6+ | Cache backend |
| **Docker & Docker Compose** | latest | Container deployment |

---

## Quick Start — Local Development

### 1. Clone the repository

```bash
git clone https://github.com/Dlahiru41/markov-rl-api-cache.git
cd markov-rl-api-cache
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
```

### 3. Start Redis

Redis must be running for caching to work. If you don't have Redis installed
locally you can start it in a container:

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

### 4. Create your `.env` file

```bash
cp .env.example .env
```

Open `.env` and set `UPSTREAM_URL` to the service you want to proxy:

```dotenv
# .env
REDIS_HOST=localhost
REDIS_PORT=6379

# ── Gateway settings ──────────────────────────────
UPSTREAM_URL=http://localhost:9000   # ← your real microservice
CACHE_DEFAULT_TTL=300                # seconds
GATEWAY_PORT=8000
CACHE_ENABLED=true
UPSTREAM_TIMEOUT_MS=5000
```

> **`UPSTREAM_URL` is the only setting you _must_ change.** Everything else has
> sensible defaults.

### 5. Start the gateway

```bash
python -m uvicorn src.gateway.proxy:create_gateway_app \
    --factory --host 0.0.0.0 --port 8000
```

The gateway is now listening on **http://localhost:8000**. Every request is
forwarded to your upstream service, and GET responses are transparently cached.

---

## Quick Start — Docker Compose

### Minimal — gateway + Redis only

If you just want the caching gateway in front of your own service:

```bash
cd docker

# Edit the UPSTREAM_URL in docker-compose.yml (markov-cache-gateway service)
# to point at your service, then:
docker compose up -d redis markov-cache-gateway
```

The gateway will be available on **http://localhost:8000**.

### Full stack — all e-commerce simulator services + monitoring

```bash
cd docker
docker compose up -d            # builds everything, starts all services
```

Or use the helper script:

```bash
cd docker
chmod +x scripts/*.sh           # first time only
./scripts/deploy_simulator.sh
```

This starts:

| Service | Port | URL |
|---------|------|-----|
| **Cache Gateway** | 8000 | http://localhost:8000 |
| Markov-RL Metrics | 9200 | http://localhost:9200/metrics |
| Auth Service | 8002 | http://localhost:8002 |
| User Service | 8001 | http://localhost:8001 |
| Product Service | 8003 | http://localhost:8003 |
| Cart Service | 8004 | http://localhost:8004 |
| Order Service | 8005 | http://localhost:8005 |
| Payment Service | 8006 | http://localhost:8006 |
| Inventory Service | 8007 | http://localhost:8007 |
| Redis | 6379 | redis://localhost:6379 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 (admin / admin) |

### Start traffic generation (optional)

```bash
cd docker
./scripts/start_traffic.sh normal    # profiles: normal, peak, degraded, burst
```

### Stop everything

```bash
cd docker
./scripts/stop_all.sh           # add --clean to also remove volumes
```

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_URL` | `http://localhost:9000` | The real service to proxy |
| `CACHE_DEFAULT_TTL` | `300` | Default cache TTL in seconds |
| `GATEWAY_PORT` | `8000` | Port the gateway listens on |
| `CACHE_ENABLED` | `true` | Enable/disable caching (`true`/`false`) |
| `UPSTREAM_TIMEOUT_MS` | `5000` | Max time to wait for upstream (ms) |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis database number |

### Per-path cache rules (`configs/cache_rules.yaml`)

You can override TTL and caching behaviour per path:

```yaml
cache_rules:
  - path: "/api/products*"
    ttl: 600
    vary_by_user: false

  - path: "/api/users/*"
    ttl: 120
    vary_by_user: true

  - path: "/api/orders/*"
    ttl: 60
    vary_by_user: true

  - path: "/api/auth/*"
    ttl: 0
    cache: false          # never cache auth endpoints
```

### YAML config file (`configs/default.yaml`)

The `gateway` section mirrors the environment variables and is used when
environment variables are not set:

```yaml
gateway:
  upstream_url: "http://localhost:9000"
  cache_default_ttl: 300
  gateway_port: 8000
  cache_enabled: true
  upstream_timeout_ms: 5000
```

---

## Verifying the Service

### 1. Health check

```bash
curl http://localhost:8000/admin/health
```

Expected response:

```json
{
  "status": "healthy",
  "upstream": { "url": "http://localhost:9000", "reachable": true },
  "redis": { "connected": true },
  "rl_agent": { "loaded": true }
}
```

### 2. Make a request through the gateway

```bash
# First request — cache MISS (forwarded to upstream)
curl -i http://localhost:8000/api/users/123
# Look for:  X-Cache: MISS

# Second request — cache HIT (served from Redis)
curl -i http://localhost:8000/api/users/123
# Look for:  X-Cache: HIT
```

### 3. Check cache statistics

```bash
curl http://localhost:8000/admin/stats
```

```json
{
  "total_requests": 2,
  "cache_hits": 1,
  "cache_misses": 1,
  "hit_rate": 0.5,
  "miss_rate": 0.5,
  "upstream_errors": 0,
  "prefetch_issued": 0,
  "prefetch_used": 0
}
```

---

## Admin Endpoints

These endpoints are handled by the gateway itself and are **not** forwarded to
upstream.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/admin/health` | Upstream, Redis, and RL agent status |
| `GET` | `/admin/stats` | Hit/miss rates, prefetch accuracy |
| `POST` | `/admin/cache/flush` | Clear all cached responses |
| `POST` | `/admin/cache/invalidate` | Invalidate keys matching a pattern |
| `GET` | `/admin/config` | Show current gateway configuration |

**Example — invalidate all user caches:**

```bash
curl -X POST http://localhost:8000/admin/cache/invalidate \
     -H "Content-Type: application/json" \
     -d '{"pattern": "/api/users/*"}'
```

---

## Running Tests

```bash
# Run all tests
python -m pytest

# Run only the gateway tests
python -m pytest tests/unit/test_gateway.py -v

# Run with verbose output
python -m pytest -xvs
```

---

## Architecture Overview

```
┌──────────┐       ┌─────────────────────────┐       ┌────────────┐
│  Client  │──────►│  markov-rl-cache:8000   │──────►│  Upstream  │
│          │◄──────│  (gateway / proxy)       │◄──────│  :9000     │
└──────────┘       └──────────┬──────────────┘       └────────────┘
                              │
                   ┌──────────▼──────────┐
                   │      Redis          │
                   │  (response cache)   │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Markov Chain +     │
                   │  DQN RL Agent       │
                   │  (async prefetch)   │
                   └─────────────────────┘
```

**GET requests** — checked against Redis; on HIT the response is returned
immediately (upstream is never called); on MISS the request is forwarded,
the response is cached, and the Markov/RL layer records the access and may
prefetch predicted next paths in the background.

**Mutation requests** (POST / PUT / PATCH / DELETE) — forwarded directly to
upstream; on a 2xx response related GET cache keys are automatically
invalidated by path prefix.

**Resilience** — if Redis is down the gateway silently bypasses caching and
proxies directly. If the upstream is unreachable the gateway returns `502 Bad
Gateway`; if the upstream is too slow it returns `504 Gateway Timeout`.

### Key source files

| File | Purpose |
|------|---------|
| `src/gateway/proxy.py` | Reverse proxy core + admin endpoints |
| `src/gateway/cache_keys.py` | Cache key builder & invalidation logic |
| `configs/cache_rules.yaml` | Per-path TTL / caching rules |
| `configs/default.yaml` | Full YAML configuration |
| `src/markov/` | Markov chain prediction models |
| `src/rl/` | DQN reinforcement learning agent |
| `src/monitoring/` | Prometheus metrics exporter |
| `docker/docker-compose.yml` | Docker Compose orchestration |
| `docker/Dockerfile` | Multi-stage Docker build |

---

## Troubleshooting

### Gateway starts but all requests return 502

Your upstream service is not reachable. Check:

```bash
curl http://localhost:9000/    # can you reach it directly?
```

Make sure `UPSTREAM_URL` in your `.env` or `docker-compose.yml` is correct and
that the upstream service is running.

### Redis connection refused

```bash
redis-cli ping    # should return PONG
```

If Redis is not running, start it:

```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

The gateway will still work without Redis — it just won't cache anything.

### Port 8000 already in use

Either stop the process occupying port 8000 or change `GATEWAY_PORT`:

```bash
GATEWAY_PORT=8080 python -m uvicorn src.gateway.proxy:create_gateway_app \
    --factory --host 0.0.0.0 --port 8080
```

### Docker build fails

```bash
cd docker
docker compose build --no-cache
```

Make sure Docker has at least 8 GB of memory allocated (Docker Desktop →
Settings → Resources → Memory).

---

## Additional Documentation

- **Full docs index:** [`docs/README.md`](docs/README.md)
- **Docker quickstart:** [`docs/deployment/docker_QUICKSTART.md`](docs/deployment/docker_QUICKSTART.md)
- **Gym environment setup:** [`docs/guides/SETUP_GUIDE.md`](docs/guides/SETUP_GUIDE.md)
- **Monitoring dashboards:** [`docs/deployment/docker_monitoring_README.md`](docs/deployment/docker_monitoring_README.md)

## API Simulation Comparison (With vs Without Solution)

Run the A/B simulation and generate comparison reports:

```powershell
python scripts/api_simulation_compare.py
```

Use a trained DQN checkpoint (optional):

```powershell
python scripts/api_simulation_compare.py --agent-model test_agent.pt
```

Outputs are written to `results/api_simulation_comparison/`:
- `comparison.json`
- `comparison.md`
- `without_solution.prom`
- `with_solution.prom`

See `docs/API_SIMULATION_COMPARISON.md` for details.
