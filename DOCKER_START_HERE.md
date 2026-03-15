# 🐳 DOCKER-ONLY DEPLOYMENT - START HERE
## Markov RL API Cache Local Setup

**Updated**: March 15, 2026  
**Scope**: Docker-based development & testing ONLY  
**No Kubernetes, No Production complexity - Just Local Docker**

---

## ✨ What You're Getting

A complete Docker-based local setup with:

✅ **Cache Intelligence Service** (Python + FastAPI)  
✅ **Redis Cache Backend** (for storage)  
✅ **Prometheus** (for metrics collection)  
✅ **Grafana** (for dashboards)  
✅ **Training Tools** (for Markov & DQN models)  
✅ **Testing Framework** (validation & integration tests)  

All containerized. All local. No production complexity.

---

## ⚡ 5-Minute Quick Start

```bash
# 1. Clone & navigate
git clone <repo-url>
cd markov-rl-api-cache/docker

# 2. Build Docker image (first time only - takes 3-5 minutes)
docker-compose build

# 3. Start services
docker-compose up -d

# 4. Verify it's running
docker-compose ps
curl http://localhost:8000/health

# 5. Done! Services available at:
# - Cache API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
# - Redis: localhost:6379
```

---

## 📚 Complete Documentation

### For Different Needs:

**"I want to setup Docker quickly"**
→ Read: DOCKER_QUICKSTART_7DAYS.md (7-day day-by-day guide)

**"I need detailed Docker setup instructions"**
→ Read: DOCKER_SETUP_GUIDE.md (complete reference)

**"I want to integrate with my application"**
→ Read: IMPLEMENTATION_REFERENCE.md (code examples)

**"I need to understand the API"**
→ Read: INTEGRATION_GUIDE.md (REST API reference)

---

## 📋 7-Day Timeline

```
DAY 1-2: Setup environment & build image
DAY 3: Start all Docker services
DAY 4: Prepare training data
DAY 5: Train Markov predictor
DAY 6: Train DQN agent
DAY 7: Test & validate everything

Result: Fully operational local environment ✅
```

See DOCKER_QUICKSTART_7DAYS.md for detailed daily tasks.

---

## 🎯 Common Tasks

### Check Service Status
```bash
docker-compose ps
# Shows all running containers

docker-compose logs -f cache-intelligence
# Shows real-time logs
```

### Train Models
```bash
# Markov predictor (1-2 minutes)
docker exec -it simulator-rl-cache python src/markov/train.py \
  --data ../data/training/sequences.json

# DQN agent (5-10 minutes for 100 episodes)
docker exec -it simulator-rl-cache python train_rl_agents.py \
  --episodes 100
```

### Test Cache Operations
```bash
# Set a value
curl -X POST http://localhost:8000/cache/set \
  -H "Content-Type: application/json" \
  -d '{"key": "test", "data": {"value": 42}, "ttl_seconds": 3600}'

# Get the value
curl http://localhost:8000/cache/get?key=test

# Get stats
curl http://localhost:8000/cache/stats
```

### View Metrics
```bash
# Open Grafana dashboard
http://localhost:3000
# Login: admin / admin
# View "Markov RL Cache Intelligence" dashboard
```

### Stop Services
```bash
docker-compose down
# Services stopped, volumes preserved

docker-compose down -v
# Completely clean, removes all data
```

---

## 🛠️ Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| Port already in use | Change port in docker-compose.yml or kill existing process |
| Out of memory | Increase Docker memory limit or stop Grafana |
| Services won't start | Check logs: `docker-compose logs` |
| Redis connection failed | Restart: `docker-compose restart redis` |
| Model training fails | Check data format: `python -m json.tool data.json` |
| Metrics not showing | Check Prometheus: `curl http://localhost:9090/-/healthy` |

See DOCKER_SETUP_GUIDE.md for detailed troubleshooting.

---

## 📦 What You Don't Have (Intentionally Excluded)

❌ Kubernetes manifests  
❌ Production deployment guides  
❌ Load balancing setup  
❌ Multi-node orchestration  
❌ Horizontal scaling  
❌ Production security hardening  

**Why?** You asked for Docker-only local setup. These are removed to keep it simple.

---

## 🔄 Typical Workflow

```
1. Start services
   └─ docker-compose up -d

2. Prepare data
   └─ Create training sequences

3. Train models
   └─ Markov (1-2 min)
   └─ DQN (5-10 min)

4. Test integration
   └─ Call REST API
   └─ Verify cache operations

5. Monitor metrics
   └─ View Grafana dashboards
   └─ Check Prometheus data

6. Stop services (when done)
   └─ docker-compose down
```

---

## 📊 Architecture (Simplified)

```
Your Application
       ↓
  Cache API (Port 8000)
   ├─ Markov Predictor
   ├─ DQN Agent
   └─ Cache Manager
       ↓
    Redis (Port 6379)

Monitoring:
  Prometheus (Port 9090) ← scrapes metrics
  Grafana (Port 3000) ← visualizes Prometheus data
```

---

## ✅ Setup Verification Checklist

After setup, verify with:

```bash
# 1. All services running
docker-compose ps
# Should show 4 services, all UP

# 2. Cache API responding
curl http://localhost:8000/health
# Should return {"status": "healthy"}

# 3. Redis accessible
docker exec -it simulator-redis redis-cli ping
# Should return PONG

# 4. Prometheus scraping
curl http://localhost:9090/-/healthy
# Should return 200 OK

# 5. Grafana accessible
curl http://localhost:3000
# Should return HTML (or open in browser)
```

All ✅? You're ready to use it!

---

## 🚀 Quick Integration Example

```python
# Python example - integrate with your app
import requests

cache_api = "http://localhost:8000"

# Ask cache what to do
decision = requests.post(
    f"{cache_api}/decide",
    json={"endpoint": "GET /api/products/123", "user_type": "premium"}
).json()

if decision['action'] == 'serve_from_cache':
    # Serve from cache
    response = requests.get(
        f"{cache_api}/cache/get",
        params={"key": "api:products:123"}
    )
    if response.json()['hit']:
        data = response.json()['data']
else:
    # Fetch from backend
    data = fetch_from_backend()
    
    # Cache it
    requests.post(
        f"{cache_api}/cache/set",
        json={"key": "api:products:123", "data": data, "ttl_seconds": 3600}
    )

return data
```

---

## 📞 Documentation Map

| Document | Purpose | Time |
|----------|---------|------|
| **DOCKER_QUICKSTART_7DAYS.md** | Day-by-day setup guide | 7 days |
| **DOCKER_SETUP_GUIDE.md** | Detailed reference | 2-3 hours |
| **IMPLEMENTATION_REFERENCE.md** | Code examples | 1-2 hours |
| **INTEGRATION_GUIDE.md** | API reference | 1 hour |

---

## 🎯 Success Metrics

You'll know it's working when:

✅ All Docker services running  
✅ Cache API responding to requests  
✅ Models can be trained  
✅ Metrics visible in Grafana  
✅ Integration tests passing  

---

## 🎉 Ready to Go!

Everything is Docker-based and local. No production complexity.

**Next Step**: 
1. Read DOCKER_QUICKSTART_7DAYS.md for day-by-day tasks
2. Or jump to DOCKER_SETUP_GUIDE.md for detailed reference

**Time to Setup**: 7 days following the timeline, or 1-2 hours if you know Docker

---

**Questions?** Refer to DOCKER_SETUP_GUIDE.md troubleshooting section.


