# Technical Integration Guide
## Markov RL API Cache - Commercial Integration

**Version**: 1.0  
**Status**: Complete Reference  
**Last Updated**: March 2026

---

## Quick Reference Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     YOUR APPLICATION LAYER                        │
│  (E-commerce, SaaS, Microservices, etc.)                          │
└─────────────────────────┬──────────────────────────────────────┘
                          │
                          ▼
                  ┌───────────────────┐
                  │  API Gateway/     │
                  │  Load Balancer    │
                  │  (NGINX/Kong/ALB) │
                  └─────────┬─────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
    ┌─────────────────────┐  ┌─────────────────────┐
    │  Cache Intelligence │  │  Cache Intelligence │
    │  Service (Port 8000)│  │  Service (Port 8000)│
    │  Instance #1        │  │  Instance #2        │
    └────────┬────────────┘  └────────┬────────────┘
             │                        │
             └────────────┬───────────┘
                          │
                ┌─────────▼─────────┐
                │ Redis Cluster     │
                │ (Master + Replicas)
                └─────────┬─────────┘
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │ Backend │  │ Backend │  │ Backend │
        │ Service │  │ Service │  │ Service │
        │ #1      │  │ #2      │  │ #N      │
        └─────────┘  └─────────┘  └─────────┘
            
            ▲             ▲             ▲
            │ Requests    │             │
            │ (cached)    │             │
            └─────────────┴─────────────┘
```

---

## Table of Contents

1. [System Components](#system-components)
2. [Integration Points](#integration-points)
3. [API Reference](#api-reference)
4. [Configuration Deep Dive](#configuration-deep-dive)
5. [Performance Tuning](#performance-tuning)
6. [Capacity Planning](#capacity-planning)
7. [Cost Reduction Analysis](#cost-reduction-analysis)

---

## System Components

### 1. Markov Predictor

**Purpose**: Predict next API call based on current sequence

```python
from src.markov import MarkovPredictor
import pickle

# Load pre-trained model
with open('models/markov_premium_users.pkl', 'rb') as f:
    predictor = pickle.load(f)

# Use in inference
predictor.reset_history()
predictor.observe('GET /api/products')
predictor.observe('GET /api/products/123')

# Get predictions
top_5 = predictor.predict(k=5)
# Output: [
#   ('GET /api/products/123/reviews', 0.45),
#   ('GET /api/products/123/images', 0.30),
#   ('GET /api/cart', 0.15),
#   ('GET /api/products/456', 0.08),
#   ('GET /api/checkout', 0.02)
# ]

# Get state vector for RL agent
state = predictor.get_state_vector(k=10)  # 10-dimensional state
# Use for decision making
```

**Key Properties**:
- **Order**: 1st-order (Markov), 2nd-order, or context-aware
- **History Size**: Configurable (5-20 typically)
- **Vocab Size**: Number of unique APIs
- **Accuracy**: Top-1 typically 60-80% depending on data

### 2. DQN Agent

**Purpose**: Make intelligent cache decisions (cache/evict/prefetch)

```python
from src.rl.agents import DQNAgent, DQNConfig

config = DQNConfig(
    state_size=64,           # Input from Markov predictor
    action_size=5,           # Number of cache actions
    hidden_dims=[256, 256],  # Network architecture
    learning_rate=0.0001,
    gamma=0.99               # Discount factor
)

agent = DQNAgent(config)

# Training
for episode in range(1000):
    state = markov.get_state_vector()
    action = agent.select_action(state, epsilon=0.1)
    
    # Execute action, get reward
    next_state, reward, done = env.step(action)
    
    # Train
    agent.remember(state, action, reward, next_state, done)
    if len(agent.memory) > 32:
        agent.replay(batch_size=32)

# Inference
state = markov.get_state_vector()
action = agent.select_action(state, epsilon=0)  # Deterministic
# action = 0: Cache, 1: Evict LRU, 2: Prefetch, etc.
```

### 3. Cache Manager

**Purpose**: Execute caching decisions and maintain cache state

```python
from src.cache import CacheManager, CacheManagerConfig

config = CacheManagerConfig(
    max_size_mb=1024,
    eviction_policy='adaptive',  # Uses RL agent
    ttl_default_seconds=3600,
    enable_compression=True,
    enable_statistics=True
)

cache = CacheManager(config)

# Usage
cache.set('api/products/123', response_data, ttl=3600)
cached_value = cache.get('api/products/123')

# Prefetch
cache.prefetch([
    'api/products/123/reviews',
    'api/products/123/images'
])

# Statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Memory usage: {stats['memory_used_mb']:.1f} MB")
print(f"Items cached: {stats['num_items']}")
```

### 4. Integration Controller

**Purpose**: Orchestrate all components

```python
from src.integration.controller import IntegrationController, ControllerConfig

# Configure
config = ControllerConfig(
    mode='deployment',
    markov_model_path='models/markov.pkl',
    agent_model_path='models/agent.pt',
    enable_monitoring=True,
    enable_api=True,
    api_port=8000,
    output_dir='/var/log/markov-cache'
)

# Initialize
controller = IntegrationController(config)
controller.setup()

# Train (if mode='training')
if config.mode == 'training':
    controller.train(num_episodes=1000)

# Or deploy (mode='deployment')
if config.mode == 'deployment':
    controller.serve()  # Starts API server
```

---

## Integration Points

### Point 1: Request Interception

You need to intercept requests to route through the cache intelligence:

#### Option A: Decorator Pattern

```python
from functools import wraps
from src.cache import CacheManager

cache = CacheManager()

def cached_endpoint(ttl=3600):
    """Decorator for caching endpoints."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{str(kwargs)}"
            
            # Try cache
            cached_response = cache.get(cache_key)
            if cached_response:
                return cached_response
            
            # Call original function
            result = await func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator

# Usage
@cached_endpoint(ttl=3600)
async def get_product(product_id: int):
    return await db.products.find_one({'id': product_id})
```

#### Option B: Middleware Pattern (FastAPI)

```python
from fastapi import FastAPI, Request
from src.cache import CacheManager

app = FastAPI()
cache = CacheManager()

@app.middleware("http")
async def cache_middleware(request: Request, call_next):
    # Only cache GET requests
    if request.method != "GET":
        return await call_next(request)
    
    # Generate cache key from path and query params
    cache_key = f"{request.url.path}:{request.url.query}"
    
    # Try cache
    if cached := cache.get(cache_key):
        return cached
    
    # Call endpoint
    response = await call_next(request)
    
    # Cache successful responses
    if response.status_code == 200:
        body = await response.body()
        cache.set(cache_key, body, ttl=3600)
        response.body = body
    
    return response
```

#### Option C: Sidecar Pattern (Kubernetes)

```yaml
# In your Kubernetes Pod spec
spec:
  containers:
  # Your main application
  - name: api
    image: your-app:latest
    ports:
    - containerPort: 8080
  
  # Cache intelligence sidecar
  - name: cache-intel
    image: markov-rl-cache:latest
    ports:
    - containerPort: 8000
    env:
    - name: REDIS_HOST
      value: "redis"
```

Then from your app:
```python
import httpx

cache_service = "http://localhost:8000"

async def get_data(endpoint: str):
    # Ask cache intelligence what to do
    decision = await httpx.post(
        f"{cache_service}/decide",
        json={"endpoint": endpoint}
    )
    
    action = decision.json()
    
    if action['action'] == 'serve_from_cache':
        return await httpx.get(
            f"{cache_service}/cache/get",
            params={"key": endpoint}
        )
    else:
        return await httpx.get(f"http://backend{endpoint}")
```

### Point 2: Metrics Collection

Send traffic data to train the system:

```python
from datetime import datetime
import json
import requests

class MetricsCollector:
    def __init__(self, cache_service_url='http://cache-intel:8000'):
        self.cache_service = cache_service_url
        self.buffer = []
    
    def record_request(self, 
                      endpoint: str,
                      user_id: str,
                      session_id: str,
                      user_type: str,
                      response_time_ms: float,
                      cache_hit: bool):
        """Record API request for model training."""
        
        self.buffer.append({
            'endpoint': endpoint,
            'user_id': user_id,
            'session_id': session_id,
            'user_type': user_type,
            'response_time_ms': response_time_ms,
            'cache_hit': cache_hit,
            'timestamp': datetime.now().isoformat()
        })
        
        # Flush every 1000 requests or 60 seconds
        if len(self.buffer) >= 1000:
            self.flush()
    
    def flush(self):
        """Send metrics to cache service."""
        if not self.buffer:
            return
        
        try:
            requests.post(
                f"{self.cache_service}/metrics/ingest",
                json={
                    'requests': self.buffer,
                    'batch_timestamp': datetime.now().isoformat()
                },
                timeout=5
            )
            self.buffer = []
        except Exception as e:
            print(f"Failed to flush metrics: {e}")

# Usage in FastAPI
metrics = MetricsCollector()

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    
    # Record metrics
    metrics.record_request(
        endpoint=request.url.path,
        user_id=request.headers.get('X-User-ID', 'anonymous'),
        session_id=request.headers.get('X-Session-ID', 'unknown'),
        user_type=request.headers.get('X-User-Type', 'guest'),
        response_time_ms=elapsed_ms,
        cache_hit=response.headers.get('X-Cache') == 'HIT'
    )
    
    return response
```

### Point 3: Monitoring Integration

Connect Prometheus to collect metrics:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'markov-rl-cache'
    static_configs:
      - targets: ['localhost:9200']
    metrics_path: '/metrics'
```

Then create Grafana dashboard with:
- Cache hit rate over time
- Average response latency
- Agent training progress
- Markov prediction accuracy
- Cost savings

---

## API Reference

### Core Endpoints

#### 1. **Cache Operations**

```python
# GET: Retrieve from cache
GET /cache/get?key=api/products/123

# Response
{
  "hit": true,
  "data": {...},
  "ttl_remaining_seconds": 3200
}

# POST: Store in cache
POST /cache/set
{
  "key": "api/products/123",
  "data": {...},
  "ttl_seconds": 3600
}

# DELETE: Remove from cache
DELETE /cache/delete?key=api/products/123

# GET: Get cache statistics
GET /cache/stats
{
  "total_hits": 10500,
  "total_misses": 3200,
  "hit_rate": 0.766,
  "memory_used_mb": 256.4,
  "num_items": 1230,
  "evictions_total": 45
}
```

#### 2. **Decision Making**

```python
# POST: Get cache decision
POST /decide
{
  "endpoint": "GET /api/products",
  "user_type": "premium",
  "session_history": [
    "GET /api/products",
    "GET /api/products/123"
  ]
}

# Response
{
  "action": "prefetch",
  "ttl_seconds": 3600,
  "prefetch_list": [
    "GET /api/products/123/reviews",
    "GET /api/products/123/images"
  ],
  "confidence": 0.82
}
```

#### 3. **Model Management**

```python
# GET: Model information
GET /models/info
{
  "markov": {
    "vocab_size": 45,
    "order": 1,
    "accuracy_top1": 0.72,
    "accuracy_top3": 0.88,
    "accuracy_top5": 0.95,
    "trained_on_samples": 50000
  },
  "agent": {
    "episodes_trained": 1250,
    "avg_reward": 342.5,
    "epsilon": 0.15,
    "buffer_size": 256000
  }
}

# POST: Load new model
POST /models/load
{
  "markov_path": "/models/markov_v2.pkl",
  "agent_path": "/models/agent_v2.pt"
}

# GET: Save current models
GET /models/save?output_dir=/backup/models
```

#### 4. **Training Control**

```python
# POST: Start training
POST /train/start
{
  "num_episodes": 1000,
  "batch_size": 64,
  "learning_rate": 0.0001
}

# Response
{
  "job_id": "train_20240315_143000",
  "status": "running"
}

# GET: Training status
GET /train/status/train_20240315_143000
{
  "status": "running",
  "episode": 256,
  "avg_reward": 287.3,
  "epsilon": 0.42
}

# GET: Training results
GET /train/results/train_20240315_143000
{
  "final_episode": 1000,
  "final_avg_reward": 452.1,
  "best_episode_reward": 480.0,
  "training_duration_seconds": 3600,
  "metrics": {...}
}
```

#### 5. **Health & Monitoring**

```python
# GET: Health check
GET /health
{
  "status": "healthy",
  "redis_connected": true,
  "models_loaded": true,
  "api_responsive": true
}

# GET: Detailed status
GET /status
{
  "mode": "deployment",
  "uptime_seconds": 86400,
  "version": "1.0.0",
  "component_health": {
    "markov_predictor": true,
    "rl_agent": true,
    "cache_manager": true,
    "redis_backend": true
  },
  "metrics": {
    "requests_handled": 1250000,
    "errors": 45,
    "avg_latency_ms": 23.5
  }
}

# Prometheus metrics endpoint
GET /metrics
# Outputs Prometheus format metrics
```

---

## Configuration Deep Dive

### Environment Variables

```bash
# ===== REDIS CONFIGURATION =====
REDIS_HOST=redis.internal.example.com
REDIS_PORT=6379
REDIS_PASSWORD=<strong-password>
REDIS_DB=0
REDIS_SSL=true
REDIS_SSL_CERT_PATH=/etc/ssl/certs/redis-cert.pem

# ===== CACHE CONFIGURATION =====
CACHE_MAX_SIZE_MB=2048          # Max cache size
CACHE_EVICTION_POLICY=adaptive   # or 'lru', 'lfu'
CACHE_TTL_DEFAULT=3600           # Default TTL in seconds
CACHE_COMPRESSION_ENABLED=true   # Enable compression
CACHE_STATISTICS_ENABLED=true    # Enable stat tracking

# ===== RL AGENT CONFIGURATION =====
AGENT_STATE_SIZE=64
AGENT_ACTION_SIZE=5
AGENT_HIDDEN_DIMS=256,256
AGENT_LEARNING_RATE=0.0001
AGENT_GAMMA=0.99
AGENT_EPSILON_DECAY=0.995

# ===== MARKOV CONFIGURATION =====
MARKOV_ORDER=1
MARKOV_HISTORY_SIZE=10
MARKOV_USE_CONTEXT_AWARE=false

# ===== API CONFIGURATION =====
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=8
API_TIMEOUT_SECONDS=30
API_ENABLE_CORS=true

# ===== MONITORING =====
PROMETHEUS_PORT=9200
ENABLE_METRICS=true
METRICS_RETENTION_DAYS=30
LOG_LEVEL=INFO

# ===== MODEL PATHS =====
MARKOV_MODEL_PATH=/models/markov_predictor.pkl
AGENT_MODEL_PATH=/models/dqn_agent.pt
TRAINING_ENABLED=false

# ===== SECURITY =====
JWT_SECRET=<generate-strong-secret>
JWT_ALGORITHM=HS256
ENABLE_AUTHENTICATION=false
RATE_LIMIT_PER_SECOND=1000
```

### YAML Configuration File

```yaml
# config.yaml
markov_rl_cache:
  # Mode: training, evaluation, deployment, demo
  mode: deployment
  
  # Redis backend
  redis:
    host: redis.production.svc.cluster.local
    port: 6379
    password_file: /run/secrets/redis_password
    ssl: true
    db: 0
  
  # Cache configuration
  cache:
    max_size_mb: 2048
    eviction_policy: adaptive  # Uses RL agent
    ttl_default_seconds: 3600
    compression_enabled: true
    statistics_enabled: true
    # Per user type configuration
    per_user_type:
      guest:
        ttl_seconds: 1800
        max_size_mb: 256
      free:
        ttl_seconds: 3600
        max_size_mb: 512
      premium:
        ttl_seconds: 7200
        max_size_mb: 1024
  
  # Markov predictor
  markov:
    order: 1  # 1st-order Markov chain
    history_size: 10
    use_context_aware: false
    min_vocab_size: 5
    
    # Per user type models
    models:
      guest: /models/markov_guest.pkl
      free: /models/markov_free.pkl
      premium: /models/markov_premium.pkl
  
  # RL Agent (DQN)
  rl_agent:
    type: dqn  # Deep Q-Network
    state_size: 64
    action_size: 5
    
    network:
      hidden_dims: [256, 256, 128]
      activation: relu
      dropout: 0.1
    
    training:
      learning_rate: 0.0001
      batch_size: 128
      gamma: 0.99  # Discount factor
      epsilon_start: 1.0
      epsilon_end: 0.01
      epsilon_decay: 0.995
      target_update_frequency: 5000
      tau: 0.005  # Soft update weight
      buffer_size: 500000
      gradient_clip: 10.0
    
    models:
      guest: /models/agent_guest.pt
      free: /models/agent_free.pt
      premium: /models/agent_premium.pt
  
  # API Gateway
  api:
    host: 0.0.0.0
    port: 8000
    workers: 8
    timeout_seconds: 30
    enable_docs: true
    enable_cors: true
    cors_origins:
      - "https://example.com"
      - "https://admin.example.com"
  
  # Monitoring
  monitoring:
    enabled: true
    prometheus_port: 9200
    metrics_retention_days: 30
    
    dashboards:
      - name: cache_performance
        interval: 1m
      - name: agent_training
        interval: 5m
      - name: system_health
        interval: 30s
    
    alerts:
      - name: low_cache_hit_rate
        threshold: 0.5
      - name: high_latency
        threshold: 1000  # ms
      - name: cascade_risk
        threshold: 0.7
  
  # Logging
  logging:
    level: INFO
    format: json  # or 'text'
    destinations:
      - stdout
      - file:/var/log/markov-cache/app.log
    
    # Don't log sensitive data
    redact_fields:
      - user_id
      - session_id
      - api_key
```

---

## Performance Tuning

### Latency Optimization

```python
# 1. Optimize state vector size
# Smaller state = faster inference
config.markov_history_size = 5  # Reduce from 10
state_size = 32  # Reduce from 64

# 2. Use model quantization
import torch
quantized_model = torch.quantization.quantize_dynamic(
    agent.model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
agent.model = quantized_model

# 3. Batch requests
# Instead of deciding per request, batch process
cache_decisions = await asyncio.gather(
    *[decide_cache_action(req) for req in batch_requests],
    return_exceptions=True
)

# 4. Cache predictions (memorization)
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_cached_prediction(endpoint_sequence):
    return markov.predict(endpoint_sequence)
```

### Throughput Optimization

```python
# 1. Increase worker processes
API_WORKERS = 16  # More cores = higher throughput

# 2. Connection pooling
from httpx import AsyncClient

# Create connection pool once
http_client = AsyncClient(limits=Limits(max_connections=100))

# 3. Batch Redis operations
pipeline = redis_client.pipeline()
for key, value in items:
    pipeline.set(key, value)
pipeline.execute()

# 4. Use async everywhere
async def process_batch(requests):
    tasks = [process_request(r) for r in requests]
    return await asyncio.gather(*tasks)
```

### Memory Optimization

```python
# 1. Limit cache size
config.cache_max_size_mb = 1024  # Hard limit

# 2. Enable compression
config.cache_compression_enabled = True

# 3. Reduce replay buffer size
config.agent_config.buffer_size = 100000  # From 500000

# 4. Use memory mapping for large models
import mmap

with open('models/large_model.bin', 'rb') as f:
    mmapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    # Use mmapped data
```

---

## Capacity Planning

### Load Estimation

```
For your system:
├─ Requests/second: ?
├─ Average response size: ? KB
├─ User types: ?
└─ Geographic regions: ?

Cache Intelligence Capacity:
├─ Per instance:
│  ├─ Throughput: 5,000-10,000 req/s (single CPU core)
│  ├─ Memory: 8-16 GB
│  ├─ Storage: 10 GB
│  └─ Cost: $100-300/month
│
├─ Scaling model:
│  ├─ 100 req/s → 1 instance
│  ├─ 1,000 req/s → 2-3 instances
│  ├─ 10,000 req/s → 5-10 instances
│  └─ 100,000 req/s → 20-50 instances
│
└─ Infrastructure:
   ├─ Redis cluster: 3+ nodes for HA
   ├─ Load balancer: 1 (or managed service)
   └─ Monitoring: Prometheus + Grafana
```

### Resource Calculation

```python
def calculate_resources(daily_requests, avg_response_kb, cache_hit_rate=0.7):
    """Calculate required infrastructure."""
    
    # Peak load (assume 3x average)
    peak_rps = (daily_requests / 86400) * 3
    
    # Instances needed (5,000 rps per instance)
    instances_needed = max(1, ceil(peak_rps / 5000))
    
    # Redis memory needed
    # Hit rate * daily_requests * response_kb
    cache_size_gb = (
        (daily_requests * cache_hit_rate * avg_response_kb) / 1024 / 1024
    )
    # Add 30% headroom
    cache_size_gb *= 1.3
    
    # Monthly cost estimate
    cost = {
        'cache_instances': instances_needed * 250,  # $250/instance
        'redis_cluster': max(500, cache_size_gb * 50),  # $50/GB
        'load_balancer': 100,
        'monitoring': 50,
        'storage': 50,
    }
    
    return {
        'instances': instances_needed,
        'cache_size_gb': cache_size_gb,
        'monthly_cost': sum(cost.values()),
        'breakdown': cost
    }

# Example
resources = calculate_resources(
    daily_requests=100_000_000,
    avg_response_kb=50,
    cache_hit_rate=0.7
)

print(f"Instances: {resources['instances']}")
print(f"Cache: {resources['cache_size_gb']:.1f} GB")
print(f"Cost: ${resources['monthly_cost']:,.0f}/month")
```

---

## Cost Reduction Analysis

### ROI Calculation

```python
def calculate_roi(
    daily_requests,
    avg_backend_cost_per_request,  # $0.001
    improved_cache_hit_rate=0.70,  # vs 0.30 with LRU
    intelligence_cost_monthly=2000
):
    """Calculate cost savings and ROI."""
    
    # Backend savings from increased cache hits
    baseline_hits = daily_requests * 0.30  # Baseline LRU hit rate
    improved_hits = daily_requests * improved_cache_hit_rate
    additional_cached_requests = improved_hits - baseline_hits
    
    monthly_backend_savings = (
        additional_cached_requests * 30 * avg_backend_cost_per_request
    )
    
    # Additional savings from request rate reduction (cascade prevention)
    cascade_reduction = 0.15  # 15% fewer downstream requests
    cascade_savings = (
        daily_requests * 30 * cascade_reduction * avg_backend_cost_per_request
    )
    
    # Total savings
    total_monthly_savings = monthly_backend_savings + cascade_savings
    net_savings = total_monthly_savings - intelligence_cost_monthly
    
    # ROI
    roi_percent = (net_savings / total_monthly_savings) * 100
    payback_months = intelligence_cost_monthly / net_savings if net_savings > 0 else float('inf')
    
    return {
        'baseline_cost_savings': monthly_backend_savings,
        'cascade_prevention_savings': cascade_savings,
        'total_backend_savings': total_monthly_savings,
        'intelligence_cost': intelligence_cost_monthly,
        'net_monthly_savings': net_savings,
        'roi_percent': roi_percent,
        'payback_months': payback_months
    }

# Example
analysis = calculate_roi(
    daily_requests=100_000_000,
    avg_backend_cost_per_request=0.0005,
    improved_cache_hit_rate=0.70,
    intelligence_cost_monthly=2000
)

print("=== FINANCIAL ANALYSIS ===")
print(f"Backend savings: ${analysis['total_backend_savings']:,.0f}/month")
print(f"Cascade prevention: ${analysis['cascade_prevention_savings']:,.0f}/month")
print(f"Intelligence cost: ${analysis['intelligence_cost']:,.0f}/month")
print(f"NET SAVINGS: ${analysis['net_monthly_savings']:,.0f}/month")
print(f"ROI: {analysis['roi_percent']:.1f}%")
print(f"Payback period: {analysis['payback_months']:.1f} months")
```

### Example ROI Scenarios

```
SCENARIO 1: E-commerce (100M requests/day)
├─ Baseline backend cost: $50,000/day
├─ Baseline cache: LRU (30% hit rate)
├─ With Markov RL: 70% hit rate
│
├─ Additional cached requests: 40M/day
├─ Daily savings: $20,000
├─ Monthly savings: $600,000
│
├─ Intelligence cost: $2,000/month
├─ Net monthly benefit: $598,000
├─ ROI: 29,900%
└─ Payback period: 4 hours

SCENARIO 2: SaaS API (10M requests/day)
├─ Baseline backend cost: $5,000/day
├─ Additional cached: 4M/day
├─ Daily savings: $2,000
├─ Monthly savings: $60,000
│
├─ Intelligence cost: $2,000/month
├─ Net monthly benefit: $58,000
├─ ROI: 2,900%
└─ Payback period: 1.7 days

SCENARIO 3: Financial Services (1M requests/day)
├─ Baseline backend cost: $10,000/day (higher margin)
├─ Additional cached: 0.4M/day
├─ Daily savings: $4,000
├─ Monthly savings: $120,000
│
├─ Intelligence cost: $2,000/month
├─ Net monthly benefit: $118,000
├─ ROI: 5,900%
└─ Payback period: 16 hours
```

---

## Common Integration Patterns

### Pattern 1: API Gateway Wrapper

```python
# gateway.py - Wraps all API calls through cache intelligence

from fastapi import FastAPI, Request
import httpx
import time

app = FastAPI()

CACHE_SERVICE = "http://cache-intel:8000"
BACKEND_SERVICES = {
    'products': 'http://products-service:8080',
    'orders': 'http://orders-service:8080',
    'users': 'http://users-service:8080',
}

async def get_cached_or_fetch(endpoint: str, user_type: str):
    """Get from cache or backend."""
    
    async with httpx.AsyncClient() as client:
        # Ask cache intelligence
        decision = await client.post(
            f"{CACHE_SERVICE}/decide",
            json={
                'endpoint': endpoint,
                'user_type': user_type
            }
        )
        action = decision.json()
        
        if action['action'] == 'serve_from_cache':
            cached = await client.get(
                f"{CACHE_SERVICE}/cache/get",
                params={'key': endpoint}
            )
            if cached.status_code == 200:
                return cached.json()
        
        # Fetch from backend
        service, path = endpoint.split('/', 1)
        backend_url = BACKEND_SERVICES[service]
        
        start = time.time()
        backend_response = await client.get(f"{backend_url}/{path}")
        latency = (time.time() - start) * 1000
        
        # Cache the response
        await client.post(
            f"{CACHE_SERVICE}/cache/set",
            json={
                'key': endpoint,
                'data': backend_response.json(),
                'ttl_seconds': action.get('ttl_seconds', 3600)
            }
        )
        
        # Prefetch if recommended
        if 'prefetch_list' in action:
            for prefetch_endpoint in action['prefetch_list']:
                # Background prefetch
                asyncio.create_task(
                    get_cached_or_fetch(prefetch_endpoint, user_type)
                )
        
        return backend_response.json()

@app.get("/api/{service}/{path:path}")
async def api_endpoint(service: str, path: str, request: Request):
    endpoint = f"{service}/{path}"
    user_type = request.headers.get('X-User-Type', 'guest')
    
    return await get_cached_or_fetch(endpoint, user_type)
```

### Pattern 2: Service Mesh Integration

```yaml
# istio-virtualservice.yaml

apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: api-gateway-vs
spec:
  hosts:
  - api-gateway
  http:
  # Route through cache intelligence
  - match:
    - uri:
        prefix: /api/
    route:
    - destination:
        host: cache-intelligence
        port:
          number: 8000
      weight: 100
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
    corsPolicy:
      allowOrigins:
      - exact: "https://example.com"
      allowMethods:
      - GET
      - POST
      allowHeaders:
      - "x-user-type"
      - "x-session-id"

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: cache-intelligence-dr
spec:
  host: cache-intelligence
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 1000
      http:
        http1MaxPendingRequests: 10000
        http2MaxRequests: 10000
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
```

---

## Troubleshooting Integration

### Debug Checklist

```bash
#!/bin/bash
# integration_debug.sh

echo "=== INTEGRATION DEBUG ==="

# 1. Connectivity
echo "1. Testing connectivity..."
curl -f http://cache-intel:8000/health || echo "❌ Cache service down"
curl -f http://redis:6379 < /dev/null && echo "✓ Redis connected" || echo "❌ Redis down"

# 2. API calls
echo "2. Testing API..."
curl -X POST http://cache-intel:8000/decide \
  -H "Content-Type: application/json" \
  -d '{"endpoint": "test", "user_type": "guest"}' \
  || echo "❌ Decision API failed"

# 3. Cache operations
echo "3. Testing cache..."
curl -X POST http://cache-intel:8000/cache/set \
  -H "Content-Type: application/json" \
  -d '{"key": "test", "data": {"value": 42}, "ttl_seconds": 3600}'

curl -X GET "http://cache-intel:8000/cache/get?key=test" \
  || echo "❌ Cache get failed"

# 4. Metrics
echo "4. Checking metrics..."
curl -s http://cache-intel:9200/metrics | grep 'markov_rl' | head -5

# 5. Logs
echo "5. Recent errors..."
tail -20 /var/log/markov-cache/error.log 2>/dev/null || echo "No error log"
```

---

**End of Technical Integration Guide**


