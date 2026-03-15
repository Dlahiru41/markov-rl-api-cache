# Quick Start: Docker-Only Local Setup (7-Day Timeline)
## Markov RL API Cache - Local Development

**Scope**: Docker-based local development ONLY  
**Timeline**: 7 days to complete setup  
**Effort**: 30-40 hours  
**Target**: Full local environment ready for testing & integration

---

## Day 1-2: Environment Setup

### Day 1: Installation & Configuration

```bash
# Prerequisites check
docker --version           # Should be 20.10+
docker-compose --version  # Should be 1.29+
python --version          # Should be 3.10+ (optional)
git --version             # Should be 2.0+

# Clone repository
git clone <repo-url>
cd markov-rl-api-cache

# Create environment file
cp .env.example .env
# Edit if needed, defaults work fine

# Verify Docker setup
docker --help
docker-compose --help
```

**Checklist:**
- ✅ Docker Desktop installed & running
- ✅ Docker Compose available
- ✅ Repository cloned
- ✅ .env file created

### Day 2: Build Docker Image

```bash
# Navigate to docker directory
cd docker

# Build the image (takes 3-5 minutes)
docker-compose build

# Verify build succeeded
docker images | grep markov-rl-cache
```

**Expected output:**
```
REPOSITORY              TAG       IMAGE ID
markov-rl-cache         latest    abc123def456
```

**Checklist:**
- ✅ Docker image built successfully
- ✅ Image size ~2GB
- ✅ Python 3.10, PyTorch installed

---

## Day 3: Start Services

### Start All Services

```bash
# From docker/ directory
docker-compose up -d

# Verify all running
docker-compose ps

# Should show 4 services:
# - simulator-redis (UP)
# - simulator-prometheus (UP)
# - simulator-grafana (UP)
# - simulator-rl-cache (UP)
```

### Verify Services

```bash
# Test each service
curl http://localhost:8000/health
# Response: {"status": "healthy"}

docker exec -it simulator-redis redis-cli ping
# Response: PONG

curl http://localhost:9090/-/healthy
# Response: 200 OK
```

**Checklist:**
- ✅ All services running
- ✅ Health checks passing
- ✅ No port conflicts
- ✅ Logs show no errors

---

## Day 4: Prepare Training Data

### Collect API Sequences

```bash
# Create data directory
mkdir -p ../data/training

# Create sample training data
cat > ../data/training/sequences.json << 'EOF'
[
  {
    "user_type": "premium",
    "sequence": [
      "GET /api/products",
      "GET /api/products/123",
      "GET /api/products/123/reviews",
      "GET /api/checkout"
    ],
    "timestamp": "2026-03-15T10:00:00"
  },
  {
    "user_type": "free",
    "sequence": [
      "GET /api/products",
      "GET /api/products/456",
      "GET /api/cart",
      "POST /api/cart"
    ],
    "timestamp": "2026-03-15T10:05:00"
  }
]
EOF
```

### Validate Data Format

```bash
# Check file created
ls -lh ../data/training/sequences.json

# Validate JSON format
python -m json.tool ../data/training/sequences.json > /dev/null && echo "✓ Valid JSON"
```

**Checklist:**
- ✅ Data file created
- ✅ Valid JSON format
- ✅ At least 10 sequences
- ✅ Mix of user types (optional but recommended)

---

## Day 5: Train Markov Model

### Train Predictor

```bash
# Run training
docker exec -it simulator-rl-cache python src/markov/train.py \
  --data ../data/training/sequences.json \
  --output ../models/markov.pkl \
  --order 1 \
  --history-size 10

# Expected output:
# Training Markov predictor...
# Processing 20 sequences...
# Vocabulary size: 8 APIs
# Top-1 accuracy: 60.0%
# Top-3 accuracy: 90.0%
# Model saved: models/markov.pkl
```

### Verify Model

```bash
# Check model was created
docker exec -it simulator-rl-cache ls -lh ../models/markov.pkl

# Test model loading
docker exec -it simulator-rl-cache python -c \
  "import pickle; m = pickle.load(open('../models/markov.pkl', 'rb')); print('✓ Model loads successfully')"
```

**Checklist:**
- ✅ Model training completed
- ✅ Model file created (~1-10 MB)
- ✅ Accuracy logged (60%+ expected)
- ✅ Model loads without errors

---

## Day 6: Train DQN Agent

### Train RL Agent

```bash
# Train for 100 episodes (takes 5-10 minutes)
docker exec -it simulator-rl-cache python train_rl_agents.py \
  --episodes 100 \
  --batch-size 32 \
  --output ../models/dqn_agent.pt \
  --verbose

# Expected output:
# Starting training...
# Episode 1: reward=120.5, epsilon=0.990
# Episode 25: reward=185.3, epsilon=0.945
# Episode 50: reward=250.7, epsilon=0.900
# Episode 75: reward=312.4, epsilon=0.855
# Episode 100: reward=350.2, epsilon=0.810
# Training complete. Model saved: models/dqn_agent.pt
```

### Monitor Training Progress

```bash
# In a separate terminal, watch logs
docker-compose logs -f cache-intelligence

# Should show training progress
# Look for: episode rewards, epsilon decay, memory usage
```

**Checklist:**
- ✅ Training completes without errors
- ✅ Reward increasing over episodes
- ✅ Epsilon decaying properly
- ✅ Model file created (~5-20 MB)

---

## Day 7: Test & Validate

### Test Cache Operations

```bash
# Test cache set
curl -X POST http://localhost:8000/cache/set \
  -H "Content-Type: application/json" \
  -d '{
    "key": "test_product",
    "data": {"id": 123, "name": "Product"},
    "ttl_seconds": 3600
  }'

# Test cache get
curl http://localhost:8000/cache/get?key=test_product
# Response: {"hit": true, "data": {...}, "ttl_remaining_seconds": 3599}

# Test cache stats
curl http://localhost:8000/cache/stats
# Response shows: hits, misses, size
```

### Test Model Predictions

```bash
# Test cache decision
curl -X POST http://localhost:8000/decide \
  -H "Content-Type: application/json" \
  -d '{
    "endpoint": "GET /api/products/123",
    "user_type": "premium"
  }'

# Response example:
# {
#   "action": "serve_from_cache",
#   "ttl_seconds": 3600,
#   "prefetch_list": ["GET /api/products/123/reviews"],
#   "confidence": 0.82
# }
```

### Run Integration Tests

```bash
# Run tests
docker exec -it simulator-rl-cache pytest tests/ -v --tb=short

# Expected output:
# test_cache.py::test_set_get PASSED
# test_markov.py::test_prediction PASSED
# test_agent.py::test_inference PASSED
# ====================== 3 passed in 2.34s =======================
```

### Check Monitoring

```bash
# Open Grafana dashboard
# URL: http://localhost:3000
# Username: admin
# Password: admin

# Metrics should show:
# - Cache operations count
# - API response times
# - System health
```

**Checklist:**
- ✅ Cache operations working
- ✅ Model predictions working
- ✅ Integration tests passing
- ✅ Grafana dashboard accessible
- ✅ Metrics showing in dashboard

---

## Success Criteria

### Complete Setup When:

✅ **Services Running**
- All 4 Docker containers healthy
- No errors in logs
- All health checks passing

✅ **Models Trained**
- Markov model file exists
- DQN agent model file exists
- Both load without errors

✅ **APIs Working**
- Cache set/get operations working
- Decision making working
- Stats endpoint responding

✅ **Monitoring Active**
- Prometheus scraping metrics
- Grafana dashboard loaded
- Metrics displaying

✅ **Tests Passing**
- Unit tests: 100% passing
- Integration tests: 100% passing
- No errors in logs

---

## Troubleshooting

### Issue: Port Already in Use

```bash
# Check what's using the port
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Solution: Stop the service
# Or change port in docker-compose.yml
```

### Issue: Out of Memory

```bash
# Check memory usage
docker stats

# Solution:
# 1. Increase Docker memory limit
# 2. Reduce cache size in .env
# 3. Stop Grafana (uses 500MB)
docker-compose stop grafana
```

### Issue: Model Training Fails

```bash
# Check error
docker-compose logs cache-intelligence

# Solution:
# 1. Verify data file format: python -m json.tool data/training/sequences.json
# 2. Ensure file permissions correct
# 3. Check disk space: docker exec simulator-rl-cache df -h
```

### Issue: Services Won't Start

```bash
# Check logs
docker-compose logs

# Restart all services
docker-compose restart

# Or full reset
docker-compose down -v
docker-compose up -d
```

---

## Daily Checklist

```
DAY 1-2: SETUP
☐ Docker installed & running
☐ Repository cloned
☐ Image built successfully

DAY 3: LAUNCH
☐ All services running
☐ Health checks passing
☐ No port conflicts

DAY 4: DATA
☐ Training data collected
☐ JSON format validated
☐ Data in correct directory

DAY 5: MARKOV TRAINING
☐ Markov model trained
☐ Model file created
☐ Model loads successfully

DAY 6: DQN TRAINING
☐ DQN agent trained
☐ Model file created
☐ Training logs show progress

DAY 7: VALIDATION
☐ Cache operations working
☐ Model predictions working
☐ Tests passing
☐ Monitoring active
☐ Ready for integration
```

---

## Next Steps

Once Day 7 complete:

1. **Integration Ready**
   - Integrate with your application
   - Use REST API endpoints
   - Follow DOCKER_SETUP_GUIDE.md

2. **Further Development**
   - Fine-tune models with real data
   - Add custom endpoints
   - Extend functionality

3. **Production (Future)**
   - Consider container orchestration
   - Scale horizontally
   - Add load balancing

---

## Resources

- **Setup Details**: See DOCKER_SETUP_GUIDE.md
- **API Reference**: See INTEGRATION_GUIDE.md
- **Code Examples**: See IMPLEMENTATION_REFERENCE.md
- **Architecture**: See System Architecture section in guides

---

**Timeline: 7 days from zero to fully operational local setup**


