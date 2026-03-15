# Deep Deployment Playbook: Markov RL API Cache
## Integration into Commercial Products

**Version**: 1.0  
**Last Updated**: March 2026  
**Target Audience**: DevOps Engineers, Solution Architects, Integration Engineers

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Docker Setup Guide](#docker-setup-guide)
4. [Phase 1: Development & Testing](#phase-1-development--testing)
5. [Local Testing & Validation](#local-testing--validation)
6. [Integration with Microservices](#integration-with-microservices)
7. [Model Training & Optimization](#model-training--optimization)
8. [Monitoring & Observability](#monitoring--observability)
9. [Local Maintenance & Operations](#local-maintenance--operations)
10. [Troubleshooting Guide](#troubleshooting-guide)

---

## Executive Summary

### What is Markov RL API Cache?

This is an **intelligent adaptive API caching system** that uses:

- **Markov Chains**: To predict API call sequences and identify prefetch opportunities
- **Reinforcement Learning (DQN)**: To dynamically optimize cache policies based on real-world traffic patterns
- **Deep Learning**: To train agents that make better cache eviction/prefetch decisions than traditional LRU/TTL strategies

### Key Benefits

| Benefit | Impact |
|---------|--------|
| **Reduced Latency** | 40-60% reduction in API response times |
| **Cost Reduction** | 30-50% fewer backend API calls through intelligent prefetching |
| **Adaptive Learning** | Automatically adjusts to changing traffic patterns |
| **Zero-Touch Operation** | Minimal manual tuning required |
| **Multi-Tenant Ready** | Support for different cache policies per user segment |

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ User Request → API Gateway                                      │
│                    ↓                                             │
│           Markov Predictor (What's next?)                        │
│                    ↓                                             │
│           RL Agent Decision (Cache or prefetch?)                 │
│                    ↓                                             │
│           Cache Manager (Redis backend)                          │
│                    ↓                                             │
│           Response + Metrics → Monitoring Stack                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## System Architecture Overview

### Component Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│  (Your Microservices / API Gateway)                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                 INTELLIGENCE LAYER                               │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Markov Predictor │  │   DQN Agent      │                    │
│  │ (Sequence model) │  │  (Decision maker)│                    │
│  └─────────┬────────┘  └────────┬─────────┘                    │
│            └──────────┬──────────┘                              │
│                       ↓                                         │
│            ┌────────────────────┐                              │
│            │  Integration API   │  (FastAPI REST)              │
│            └────────────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CACHE LAYER                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ CacheManager: Policy Enforcement                          │ │
│  │ ├─ TTL Management                                          │ │
│  │ ├─ Eviction Policies (LRU, DQN-optimized)                  │ │
│  │ └─ Prefetch Rules                                          │ │
│  └───────────────┬──────────────────────────────────────────┘ │
│                  ▼                                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Redis Backend (or In-Memory)                              │ │
│  │ ├─ Persistent cache storage                                │ │
│  │ ├─ TTL handling                                            │ │
│  │ ├─ Cluster support                                         │ │
│  │ └─ High availability                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                OBSERVABILITY LAYER                               │
│  Prometheus → Grafana → Alerting                                │
│  (Real-time metrics, dashboards, alerting)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **Markov Predictor** (`src/markov/`)
- **Purpose**: Predicts next API calls based on user behavior
- **Input**: Sequence of API endpoints
- **Output**: Top-K probable next endpoints with confidence scores
- **Supported Orders**: 1st-order (Markov), 2nd-order, Context-aware

#### 2. **RL Agent (DQN)** (`src/rl/`)
- **Purpose**: Makes cache decisions (cache/evict/prefetch)
- **State**: Current cache status + Markov predictions
- **Actions**: Cache operations
- **Reward**: Latency reduction + cost savings
- **Training**: Reinforcement learning with experience replay

#### 3. **Cache Manager** (`src/cache/`)
- **Purpose**: Central cache orchestrator
- **Features**:
  - TTL-based expiration
  - Automatic prefetching
  - Multiple eviction policies
  - Per-user-type policies
  - Statistics tracking

#### 4. **Integration Controller** (`src/integration/controller.py`)
- **Purpose**: Orchestrates all components
- **Modes**: Training, Evaluation, Deployment, Demo
- **Responsibilities**:
  - Component lifecycle management
  - Training orchestration
  - API exposure
  - Monitoring setup

#### 5. **Monitoring Stack** (`src/monitoring/`)
- **Metrics**: Cache hits/misses, latency, predictions accuracy
- **Backends**: Prometheus + Grafana
- **Custom Dashboards**: Cache performance, RL training progress

---

## Pre-Deployment Planning

### 1. **Requirements Assessment**

**Compute Requirements**:
```
Minimum (Development):
- 2 CPU cores
- 4 GB RAM
- 10 GB storage

Recommended (Production):
- 8+ CPU cores (8 physical for training)
- 32 GB RAM (16 GB min)
- 100+ GB storage (SSD)

Scaling (Multiple instances):
- Load balancer (NGINX, HAProxy)
- Shared Redis cluster
- Distributed training
```

**Dependencies**:
```
Core:
- Python 3.10+ (3.11 recommended)
- Redis 7.0+ (for cache backend)
- FastAPI 0.104.0+

ML Stack:
- PyTorch 2.0.0+ (CPU or GPU variant)
- NumPy 1.24.0+
- Pandas 2.0.0+

Monitoring:
- Prometheus 2.45.0+
- Grafana 10.0.0+

Optional:
- CUDA 12.0+ (for GPU training)
- Kubernetes 1.27+ (for container orchestration)
```

### 2. **Architecture Decision Matrix**

Choose based on your scale:

| Deployment Pattern | Size | Architecture |
|--------------------|------|--------------|
| **Standalone** | <100k req/day | Single machine, in-memory cache |
| **Small HA** | 100k-1M req/day | Single machine + Redis, local training |
| **Medium Scale** | 1M-10M req/day | API gateway + Redis cluster, periodic training |
| **Enterprise** | 10M+ req/day | Multi-node deployment, distributed training |

### 3. **Commercial Integration Points**

Identify where this system integrates:

```
Your Infrastructure:
├─ API Gateway / Load Balancer (NGINX, Kong, AWS ALB)
│  └─ [Deploy Intelligence Layer here]
├─ Backend Services (Microservices)
├─ Cache Layer (Redis)
└─ Monitoring (Prometheus/Grafana)
```

### 4. **Data Collection Strategy**

Define what to track for Markov training:

```yaml
Track the following per request:
- user_id / session_id
- endpoint_called
- timestamp
- response_latency
- cache_status (hit/miss)
- user_type (guest/free/premium)
```

---

## Environment Setup

### 1. **Local Development Setup**

```powershell
# Clone and navigate
cd C:\path\to\markov-rl-api-cache

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Optional: GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Create .env file
Copy-Item .env.example .env
# Edit .env with your settings
```

**Configuration** (`.env`):
```dotenv
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# API Gateway
API_HOST=0.0.0.0
API_PORT=8000

# Monitoring
PROMETHEUS_PORT=8001
ENABLE_MONITORING=true

# Model Training
TRAINING_MODE=true
LOG_LEVEL=INFO
```

### 2. **Docker Development Environment**

```bash
# Build Docker image
docker build -f docker/Dockerfile -t markov-rl-cache:latest .

# Start development stack
cd docker
docker-compose up -d

# Verify services
docker-compose ps
```

**Services Started**:
- Redis: `localhost:6379`
- Prometheus: `localhost:9090`
- Grafana: `localhost:3000` (admin/admin)
- RL Cache API: `localhost:8000`

### 3. **Production Environment Variables**

Create `.env.production`:
```dotenv
# ============= REDIS =============
REDIS_HOST=redis-cluster.internal.example.com
REDIS_PORT=6379
REDIS_PASSWORD=<generate-strong-password>
REDIS_DB=0
REDIS_SSL=true

# ============= API =============
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=8
API_TIMEOUT=30

# ============= MONITORING =============
PROMETHEUS_PORT=9200
ENABLE_MONITORING=true
METRICS_RETENTION_DAYS=30

# ============= SECURITY =============
JWT_SECRET=<generate-strong-secret>
ENABLE_RATE_LIMITING=true
RATE_LIMIT_PER_SECOND=1000

# ============= MODEL PATHS =============
MARKOV_MODEL_PATH=/models/markov_predictor.pkl
AGENT_MODEL_PATH=/models/dqn_agent.pt
```

---

## Phase 1: Development & Testing

### Step 1.1: Validate Installation

```python
# quick_validate.py
from src.integration.controller import IntegrationController, ControllerConfig
from src.markov import MarkovPredictor
from src.cache import CacheManager

# Test Markov
print("Testing Markov Predictor...")
predictor = MarkovPredictor(order=1)
sequences = [['api1', 'api2', 'api3'], ['api1', 'api2', 'api4']]
predictor.fit(sequences)
print("✓ Markov working")

# Test Cache Manager
print("Testing Cache Manager...")
cache = CacheManager()
cache.set('test_key', 'test_value', ttl=3600)
assert cache.get('test_key') == 'test_value'
print("✓ Cache Manager working")

# Test Integration Controller
print("Testing Integration Controller...")
config = ControllerConfig(mode='demo')
controller = IntegrationController(config)
controller.setup()
print("✓ Integration Controller working")

print("\n✅ All components validated successfully")
```

Run validation:
```powershell
python quick_validate.py
```

### Step 1.2: Collect Training Data

From your production systems, collect API call sequences:

```python
# data_collector.py
import json
from datetime import datetime
from collections import defaultdict

class APISequenceCollector:
    def __init__(self):
        self.sequences = defaultdict(list)
        self.session_data = []
    
    def record_api_call(self, session_id, user_type, endpoint):
        """Record an API call."""
        self.sequences[session_id].append({
            'endpoint': endpoint,
            'timestamp': datetime.now().isoformat()
        })
    
    def finalize_session(self, session_id, user_type):
        """Mark session as complete."""
        endpoints = [e['endpoint'] for e in self.sequences[session_id]]
        self.session_data.append({
            'user_type': user_type,
            'sequence': endpoints,
            'length': len(endpoints),
            'timestamp': datetime.now().isoformat()
        })
    
    def export_training_data(self, filepath):
        """Export for Markov training."""
        with open(filepath, 'w') as f:
            json.dump(self.session_data, f, indent=2)
```

### Step 1.3: Train Markov Model

```python
# train_markov.py
from src.markov import MarkovPredictor
import json

# Load training data
with open('data/api_sequences.json') as f:
    data = json.load(f)

# Extract sequences
sequences = [item['sequence'] for item in data]

# Train model (separate for each user type)
for user_type in ['guest', 'free', 'premium']:
    user_sequences = [item['sequence'] for item in data 
                      if item['user_type'] == user_type]
    
    predictor = MarkovPredictor(order=1, history_size=10)
    predictor.fit(user_sequences)
    
    # Save model
    import pickle
    with open(f'models/markov_{user_type}.pkl', 'wb') as f:
        pickle.dump(predictor, f)
    
    print(f"✓ Trained Markov model for {user_type} users")
```

### Step 1.4: Test RL Agent Training (Local)

```python
# test_rl_training.py
from src.integration.controller import IntegrationController, ControllerConfig
from src.rl.agents import DQNConfig
from src.integration.gym_environment import CacheEnvConfig

# Configure for testing
env_config = CacheEnvConfig(
    num_apis=20,
    max_steps_per_episode=100,
    use_real_services=False
)

agent_config = DQNConfig(
    hidden_dims=[64, 64],
    learning_rate=0.0001,
    batch_size=32,
    buffer_size=10000
)

config = ControllerConfig(
    mode='training',
    env_config=env_config,
    agent_config=agent_config,
    enable_monitoring=True,
    output_dir='results/test_run'
)

# Train locally
controller = IntegrationController(config)
controller.setup()
print("Starting training...")
controller.train(num_episodes=50)

# Save models
controller.save_models('models/checkpoint_test')
print("✓ RL training completed successfully")
```

### Step 1.5: Run Integration Tests

```powershell
# Run existing test suite
pytest tests/ -v --tb=short

# Generate test report
pytest tests/ --html=test_report.html --self-contained-html
```

---

## Phase 2: Staging Deployment

### Step 2.1: Staging Infrastructure

Deploy to staging environment (mirrors production):

```yaml
# docker-compose.staging.yml
version: '3.8'

services:
  # Redis cluster (staging)
  redis-primary:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis-staging-data:/data

  # Cache Intelligence Service
  cache-intelligence:
    build:
      context: .
      dockerfile: docker/Dockerfile
    environment:
      - REDIS_HOST=redis-primary
      - REDIS_PORT=6379
      - API_PORT=8000
      - ENABLE_MONITORING=true
      - LOG_LEVEL=DEBUG
    ports:
      - "8000:8000"
      - "8001:8001"  # Prometheus metrics
    depends_on:
      - redis-primary
    volumes:
      - ./models:/models:ro
      - ./logs:/app/logs
    restart: unless-stopped

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./docker/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-staging-data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=staging123
    volumes:
      - grafana-staging-data:/var/lib/grafana
      - ./docker/monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro

volumes:
  redis-staging-data:
  prometheus-staging-data:
  grafana-staging-data:
```

Deploy:
```bash
docker-compose -f docker-compose.staging.yml up -d
```

### Step 2.2: Load Testing

Use production-like traffic:

```python
# load_test.py
import concurrent.futures
import time
import requests
from random import choice
import statistics

class StagingLoadTest:
    def __init__(self, api_endpoint='http://localhost:8000'):
        self.api = api_endpoint
        self.endpoints = [
            '/api/users/profile',
            '/api/products/search',
            '/api/products/details',
            '/api/reviews/list',
            '/api/orders/history',
            '/api/recommendations',
            '/api/checkout/validate'
        ]
        self.latencies = []
    
    def simulate_user_session(self, num_requests=50):
        """Simulate a single user session."""
        session_latencies = []
        for _ in range(num_requests):
            endpoint = choice(self.endpoints)
            start = time.time()
            try:
                response = requests.get(
                    f"{self.api}/cache/get",
                    params={'endpoint': endpoint},
                    timeout=5
                )
                latency = (time.time() - start) * 1000  # ms
                session_latencies.append(latency)
            except Exception as e:
                print(f"Error: {e}")
        return session_latencies
    
    def run_load_test(self, num_users=100, requests_per_user=50):
        """Run concurrent user simulation."""
        print(f"Starting load test: {num_users} users × {requests_per_user} requests")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = [
                executor.submit(self.simulate_user_session, requests_per_user)
                for _ in range(num_users)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                self.latencies.extend(future.result())
        
        # Calculate statistics
        results = {
            'total_requests': len(self.latencies),
            'avg_latency_ms': statistics.mean(self.latencies),
            'p50_latency_ms': statistics.median(self.latencies),
            'p95_latency_ms': sorted(self.latencies)[int(0.95 * len(self.latencies))],
            'p99_latency_ms': sorted(self.latencies)[int(0.99 * len(self.latencies))],
            'min_latency_ms': min(self.latencies),
            'max_latency_ms': max(self.latencies)
        }
        
        return results

# Run test
tester = StagingLoadTest()
results = tester.run_load_test(num_users=100, requests_per_user=100)

print("\n=== Load Test Results ===")
for key, value in results.items():
    print(f"{key}: {value:.2f}")

# With intelligent caching, you should see:
# - 40-60% latency reduction
# - Higher cache hit rates (>70%)
```

### Step 2.3: Integration Testing

Test with your actual microservices:

```python
# integration_test_gateway.py
"""
Test integration with API Gateway
"""
import asyncio
import aiohttp
from datetime import datetime

class GatewayIntegrationTest:
    def __init__(self, gateway_url='http://localhost:8000'):
        self.gateway_url = gateway_url
    
    async def test_cache_flow(self):
        """Test: Request → Cache Check → Backend → Response"""
        
        async with aiohttp.ClientSession() as session:
            # First request (cache miss)
            print("Test 1: Initial request (cache miss)...")
            async with session.get(
                f"{self.gateway_url}/api/products/123",
                headers={'User-Type': 'premium'}
            ) as resp:
                data1 = await resp.json()
                time1 = resp.headers.get('X-Response-Time')
                cached1 = resp.headers.get('X-Cache') == 'HIT'
            
            print(f"  Status: {resp.status}, Time: {time1}ms, Cached: {cached1}")
            
            # Second request (should be cached)
            print("Test 2: Repeated request (cache hit)...")
            async with session.get(
                f"{self.gateway_url}/api/products/123",
                headers={'User-Type': 'premium'}
            ) as resp:
                data2 = await resp.json()
                time2 = resp.headers.get('X-Response-Time')
                cached2 = resp.headers.get('X-Cache') == 'HIT'
            
            print(f"  Status: {resp.status}, Time: {time2}ms, Cached: {cached2}")
            
            # Verify results
            assert data1 == data2, "Response data mismatch"
            assert cached2, "Second request should be cached"
            assert float(time2) < float(time1), "Cached response should be faster"
            
            print("\n✓ Gateway integration test passed")

# Run test
asyncio.run(GatewayIntegrationTest().test_cache_flow())
```

### Step 2.4: Staging Sign-off Checklist

```markdown
## Staging Validation Checklist

- [ ] All services startup successfully
- [ ] Redis connectivity verified
- [ ] Prometheus scraping metrics
- [ ] Grafana dashboards operational
- [ ] Load test completed (latency reduction ≥40%)
- [ ] Integration test with gateway passed
- [ ] No critical errors in logs
- [ ] Memory usage stable (<80% of limit)
- [ ] CPU usage reasonable (<60% sustained)
- [ ] Cache hit rate > 70%
- [ ] No data corruption issues
- [ ] Response consistency verified
- [ ] Monitoring alerts functional
- [ ] Security scanning passed (OWASP)
- [ ] Documentation reviewed and accurate
```

---

## Phase 3: Production Deployment

### Step 3.1: Pre-Production Checklist

```yaml
Infrastructure:
  - Redis cluster (3+ nodes, HA configured)
  - Load balancer configured
  - Network security groups configured
  - SSL/TLS certificates ready
  - Backup strategy in place

Capacity Planning:
  - Peak load analysis completed
  - Auto-scaling policies defined
  - Resource limits set appropriately
  - Disaster recovery plan documented

Models:
  - Markov model trained on real data
  - DQN agent trained (minimum 500 episodes)
  - Models validated on holdout set
  - Model versioning strategy established

Monitoring:
  - Prometheus scrape configured
  - Grafana dashboards created
  - Alert rules defined and tested
  - Logging centralized (ELK/Splunk)

Security:
  - JWT authentication configured
  - Rate limiting enabled
  - DDoS protection configured
  - Encryption in transit (TLS)
  - Encryption at rest configured
  - Security audit completed

Documentation:
  - Runbook created
  - Incident response plan
  - Deployment guide
  - Architecture documented
  - API documentation current
```

### Step 3.2: Production Deployment Architecture

```
                    ┌─────────────────────┐
                    │   User Requests     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Load Balancer      │
                    │  (NGINX/HAProxy)    │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Cache-Intel  │  │ Cache-Intel  │  │ Cache-Intel  │
    │ Pod #1       │  │ Pod #2       │  │ Pod #N       │
    │ (:8000)      │  │ (:8000)      │  │ (:8000)      │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Redis Cluster  │
                    │  (3+ nodes HA)  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Redis #1     │ │ Redis #2     │ │ Redis #3     │
    │ (Primary)    │ │ (Replica)    │ │ (Replica)    │
    └──────────────┘ └──────────────┘ └──────────────┘
```

### Step 3.3: Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: markov-rl-cache
  namespace: production
  labels:
    app: markov-rl-cache
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  
  selector:
    matchLabels:
      app: markov-rl-cache
  
  template:
    metadata:
      labels:
        app: markov-rl-cache
        version: v1
    
    spec:
      serviceAccountName: markov-rl-cache
      
      # Pod anti-affinity: spread across nodes
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - markov-rl-cache
              topologyKey: kubernetes.io/hostname
      
      containers:
      - name: cache-intelligence
        image: registry.example.com/markov-rl-cache:1.0.0
        imagePullPolicy: IfNotPresent
        
        ports:
        - name: api
          containerPort: 8000
          protocol: TCP
        - name: metrics
          containerPort: 8001
          protocol: TCP
        
        env:
        - name: REDIS_HOST
          value: "redis-cluster.production.svc.cluster.local"
        - name: REDIS_PORT
          value: "6379"
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: password
        - name: API_WORKERS
          value: "4"
        - name: ENABLE_MONITORING
          value: "true"
        - name: LOG_LEVEL
          value: "INFO"
        - name: MODEL_PATH_MARKOV
          value: "/models/markov_predictor.pkl"
        - name: MODEL_PATH_AGENT
          value: "/models/dqn_agent.pt"
        
        # Resource limits
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: api
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: api
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        
        # Volume mounts
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: logs
          mountPath: /app/logs
        
        # Security context
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
      
      volumes:
      - name: models
        configMap:
          name: markov-rl-models
      - name: logs
        emptyDir: {}
      
      # Termination grace period for graceful shutdown
      terminationGracePeriodSeconds: 30

---
apiVersion: v1
kind: Service
metadata:
  name: markov-rl-cache
  namespace: production
spec:
  selector:
    app: markov-rl-cache
  type: ClusterIP
  ports:
  - name: api
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 8001
    targetPort: 8001

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: markov-rl-cache-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: markov-rl-cache
  
  minReplicas: 3
  maxReplicas: 10
  
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

Deploy:
```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl rollout status deployment/markov-rl-cache -n production
```

### Step 3.4: Production Deployment via Docker Compose

For non-Kubernetes deployments:

```yaml
# docker-compose.production.yml
version: '3.8'

networks:
  prod-network:
    driver: bridge

volumes:
  redis-prod-data:
    driver: local
  prometheus-prod-data:
    driver: local

services:
  # ===== CACHE INTELLIGENCE SERVICE =====
  cache-intel-1:
    image: markov-rl-cache:latest
    container_name: cache-intel-1
    environment:
      REDIS_HOST: redis-primary
      REDIS_PORT: 6379
      REDIS_PASSWORD_FILE: /run/secrets/redis_password
      API_HOST: 0.0.0.0
      API_PORT: 8000
      API_WORKERS: 4
      ENABLE_MONITORING: "true"
      LOG_LEVEL: INFO
    ports:
      - "8001:8000"
      - "9201:9200"
    depends_on:
      redis-primary:
        condition: service_healthy
    networks:
      - prod-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G
        reservations:
          cpus: '1'
          memory: 4G
    secrets:
      - redis_password
    volumes:
      - ./models:/models:ro
      - /var/log/markov-cache/instance1:/app/logs

  cache-intel-2:
    # ... (duplicate of cache-intel-1, but on port 8002/9202)
    image: markov-rl-cache:latest
    container_name: cache-intel-2
    environment:
      REDIS_HOST: redis-primary
      REDIS_PORT: 6379
      REDIS_PASSWORD_FILE: /run/secrets/redis_password
      API_PORT: 8000
      API_WORKERS: 4
    ports:
      - "8002:8000"
      - "9202:9200"
    depends_on:
      - redis-primary
    networks:
      - prod-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 8G
        reservations:
          cpus: '1'
          memory: 4G
    secrets:
      - redis_password
    volumes:
      - ./models:/models:ro
      - /var/log/markov-cache/instance2:/app/logs

  # ===== REDIS CLUSTER =====
  redis-primary:
    image: redis:7-alpine
    container_name: redis-primary-prod
    command: >
      redis-server
      --maxmemory 4gb
      --maxmemory-policy allkeys-lru
      --appendonly yes
      --appendfsync everysec
      --save 60 1000
      --requirepass ${REDIS_PASSWORD}
    networks:
      - prod-network
    volumes:
      - redis-prod-data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 6G
        reservations:
          cpus: '1'
          memory: 3G

  # ===== MONITORING =====
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus-prod
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-prod-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9090:9090"
    networks:
      - prod-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G

  grafana:
    image: grafana/grafana:latest
    container_name: grafana-prod
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_INSTALL_PLUGINS: redis-datasource
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
    ports:
      - "3000:3000"
    networks:
      - prod-network
    restart: unless-stopped
    depends_on:
      - prometheus

  # ===== NGINX LOAD BALANCER =====
  nginx:
    image: nginx:alpine
    container_name: nginx-lb
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    ports:
      - "80:80"
      - "443:443"
    networks:
      - prod-network
    depends_on:
      - cache-intel-1
      - cache-intel-2
    restart: unless-stopped

secrets:
  redis_password:
    file: ./secrets/redis_password.txt
```

**NGINX Configuration** (`nginx/nginx.conf`):
```nginx
upstream cache_intelligence {
    least_conn;  # Load balancing strategy
    server cache-intel-1:8000 max_fails=2 fail_timeout=10s;
    server cache-intel-2:8000 max_fails=2 fail_timeout=10s;
}

server {
    listen 443 ssl http2;
    server_name cache-api.example.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=general:10m rate=100r/s;
    limit_req zone=general burst=200 nodelay;
    
    location / {
        proxy_pass http://cache_intelligence;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    # Metrics endpoint (internal only)
    location /metrics {
        allow 10.0.0.0/8;  # Internal network
        deny all;
        proxy_pass http://cache_intelligence;
    }
}

server {
    listen 80;
    server_name cache-api.example.com;
    return 301 https://$server_name$request_uri;
}
```

Deploy production:
```bash
docker-compose -f docker-compose.production.yml up -d
docker-compose ps
```

### Step 3.5: Production Go-Live Checklist

```markdown
## Production Go-Live Verification

**24 hours before:**
- [ ] Final staging validation complete
- [ ] Runbooks distributed to team
- [ ] On-call schedule confirmed
- [ ] Rollback procedure documented and tested

**6 hours before:**
- [ ] Production environment health check
- [ ] Database backups verified
- [ ] Load balancer tested
- [ ] All monitoring dashboards accessible

**At deployment time:**
- [ ] Start with 10% traffic (canary deployment)
- [ ] Monitor metrics for 15 minutes
  - [ ] Error rate < 0.1%
  - [ ] P99 latency stable
  - [ ] Cache hit rate increasing
- [ ] Increase to 50% traffic
- [ ] Monitor for 30 minutes
- [ ] If all good, increase to 100%

**Post-deployment:**
- [ ] Metrics baseline established
- [ ] Customer feedback monitored
- [ ] Performance targets validated
- [ ] Team debriefing scheduled
```

---

## Integration with Microservices

### Step 1: API Gateway Integration

Modify your API gateway to route through the cache intelligence system:

**Option A: Sidecar Pattern (Kubernetes)**
```yaml
# In your microservice Pod spec
spec:
  containers:
  - name: api-gateway
    image: your-gateway:latest
    env:
    - name: CACHE_SERVICE_URL
      value: "http://localhost:8000"
  
  - name: cache-intelligence
    image: markov-rl-cache:latest
    ports:
    - containerPort: 8000
```

**Option B: Reverse Proxy Pattern**
```python
# In your API gateway (FastAPI example)
from fastapi import FastAPI, Request
import httpx

app = FastAPI()
cache_service = "http://cache-intelligence:8000"

@app.get("/api/{path:path}")
async def handle_api_request(path: str, request: Request):
    # Check cache intelligence
    async with httpx.AsyncClient() as client:
        # Get cache decision
        decision = await client.post(
            f"{cache_service}/decide",
            json={
                "endpoint": path,
                "user_type": request.headers.get("X-User-Type"),
                "session_id": request.headers.get("X-Session-ID")
            }
        )
        
        cache_action = decision.json()
        
        if cache_action["action"] == "serve_from_cache":
            # Fetch from cache
            cached = await client.get(
                f"{cache_service}/cache/get",
                params={"key": path}
            )
            return cached.json()
        
        elif cache_action["action"] == "prefetch":
            # Prefetch related APIs
            for api in cache_action["prefetch_list"]:
                # Async request to backend
                pass
        
        # Fall through to backend
        backend_response = await client.get(f"http://backend/{path}")
        
        # Cache the response
        await client.post(
            f"{cache_service}/cache/set",
            json={
                "key": path,
                "value": backend_response.json(),
                "ttl": cache_action.get("ttl", 3600)
            }
        )
        
        return backend_response.json()
```

### Step 2: Metrics Integration

Send real traffic metrics to the Markov predictor:

```python
# In your API gateway
from datetime import datetime

class MetricsCollector:
    def __init__(self, cache_service_url):
        self.cache_service = cache_service_url
        self.buffer = []
    
    def record_request(self, endpoint, user_id, session_id, user_type, latency_ms):
        """Record API request."""
        self.buffer.append({
            'endpoint': endpoint,
            'user_id': user_id,
            'session_id': session_id,
            'user_type': user_type,
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat()
        })
        
        # Flush periodically (every 100 requests or 60 seconds)
        if len(self.buffer) >= 100:
            self.flush()
    
    def flush(self):
        """Send buffered metrics to cache intelligence."""
        if not self.buffer:
            return
        
        requests.post(
            f"{self.cache_service}/metrics/ingest",
            json={"requests": self.buffer},
            timeout=5
        )
        self.buffer = []
```

### Step 3: Service Mesh Integration (Istio)

If using Istio, define VirtualService to route through cache:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: api-gateway
spec:
  hosts:
  - api-gateway
  http:
  # Route to cache intelligence first
  - match:
    - uri:
        prefix: /api/
    route:
    - destination:
        host: markov-rl-cache
        port:
          number: 8000
      weight: 100
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s

# Service entry for backend services
---
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: backend-services
spec:
  hosts:
  - "*.backend.svc.cluster.local"
  ports:
  - number: 8080
    name: http
    protocol: HTTP
```

---

## Model Training & Optimization

### Phase 1: Initial Model Training

```python
# train_initial_models.py
"""
Training pipeline for production models.
"""

from src.integration.controller import IntegrationController, ControllerConfig
from src.rl.agents import DQNConfig
from src.integration.gym_environment import CacheEnvConfig
from src.markov import MarkovPredictor
import json
import pickle

# ===== STEP 1: TRAIN MARKOV MODELS =====
print("=" * 70)
print("TRAINING MARKOV PREDICTOR MODELS")
print("=" * 70)

# Load production API sequence data
with open('data/production_sequences.json') as f:
    sequences_data = json.load(f)

# Train per user type
markov_models = {}
for user_type in ['guest', 'free', 'premium']:
    print(f"\nTraining Markov for {user_type} users...")
    
    user_sequences = [
        item['sequence'] 
        for item in sequences_data 
        if item['user_type'] == user_type
    ]
    
    if not user_sequences:
        print(f"  ⚠️  No data for {user_type}")
        continue
    
    # Create and train model
    predictor = MarkovPredictor(order=1, history_size=10)
    predictor.fit(user_sequences)
    
    # Evaluate
    accuracy = predictor.evaluate(user_sequences)
    print(f"  ✓ Trained on {len(user_sequences)} sequences")
    print(f"  ✓ Top-1 accuracy: {accuracy['top_1']:.2%}")
    print(f"  ✓ Vocab size: {predictor.vocab_size}")
    
    # Save
    markov_models[user_type] = predictor
    with open(f'models/markov_{user_type}.pkl', 'wb') as f:
        pickle.dump(predictor, f)

# ===== STEP 2: TRAIN DQN AGENTS =====
print("\n" + "=" * 70)
print("TRAINING DQN AGENTS")
print("=" * 70)

# Configuration
env_config = CacheEnvConfig(
    num_apis=50,  # Realistic number
    user_types=['guest', 'free', 'premium'],
    max_steps_per_episode=500,
    use_real_services=False,  # Use simulator
    cascade_threshold=0.8
)

agent_config = DQNConfig(
    hidden_dims=[256, 256, 128],
    learning_rate=0.0001,
    batch_size=128,
    buffer_size=500000,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    target_update_frequency=5000,
    tau=0.005,
    gamma=0.99
)

training_config = {
    'num_episodes': 2000,
    'eval_frequency': 100,
    'save_frequency': 200,
    'early_stopping': True,
    'patience': 300,
    'min_improvement': 0.01
}

# Train controller
config = ControllerConfig(
    mode='training',
    env_config=env_config,
    agent_config=agent_config,
    enable_monitoring=True,
    output_dir='models/training_run'
)

controller = IntegrationController(config)
controller.setup()

print("\nStarting DQN training...")
print("This may take 1-2 hours on CPU, 15-30 minutes on GPU")

controller.train(
    num_episodes=training_config['num_episodes'],
    eval_frequency=training_config['eval_frequency']
)

# ===== STEP 3: EVALUATE MODELS =====
print("\n" + "=" * 70)
print("EVALUATING MODELS")
print("=" * 70)

evaluation_results = controller.evaluate(num_episodes=100)

print("\nEvaluation Results:")
print(f"  Average Reward: {evaluation_results['avg_reward']:.2f}")
print(f"  Average Latency: {evaluation_results['avg_latency']:.2f}ms")
print(f"  Cache Hit Rate: {evaluation_results['cache_hit_rate']:.2%}")
print(f"  Cost Savings: {evaluation_results['cost_savings']:.2%}")

# ===== STEP 4: SAVE PRODUCTION MODELS =====
print("\nSaving production models...")
controller.save_models('models/production_v1.0')
print("✓ Models saved to models/production_v1.0/")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
```

Run training:
```bash
python train_initial_models.py 2>&1 | tee training_log.txt
```

### Phase 2: Continuous Model Updates

In production, retrain periodically with new data:

```python
# continuous_training.py
"""
Scheduled training job for continuous improvement.
Runs daily/weekly to adapt to changing traffic patterns.
"""

import schedule
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
from src.integration.controller import IntegrationController, ControllerConfig

class ContinuousTrainingManager:
    def __init__(self, config_path='configs/training.yaml', 
                 data_dir='data/production',
                 model_dir='models/versioned'):
        self.config_path = config_path
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_training_data(self):
        """Collect 24 hours of production data."""
        # In production, this would query your metrics system
        # (Prometheus, ELK, CloudWatch, etc.)
        cutoff_time = datetime.now() - timedelta(days=1)
        
        # Aggregate sequences from your monitoring
        sequences = []  # Load from your metrics backend
        
        return sequences
    
    def retrain_models(self):
        """Retrain both Markov and DQN models."""
        
        print(f"\n{'='*70}")
        print(f"Continuous Model Retraining - {datetime.now()}")
        print(f"{'='*70}")
        
        try:
            # Collect recent data
            print("1. Collecting production data...")
            sequences = self.collect_training_data()
            print(f"   ✓ Collected {len(sequences)} sequences")
            
            # Create timestamped directory for this run
            run_dir = self.model_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Train models
            print("2. Training models...")
            config = ControllerConfig(
                mode='training',
                output_dir=str(run_dir),
                enable_monitoring=True
            )
            
            controller = IntegrationController(config)
            controller.setup()
            controller.train(num_episodes=500)  # Smaller training for updates
            
            # Evaluate
            print("3. Evaluating new models...")
            results = controller.evaluate(num_episodes=50)
            
            # Compare with current production model
            print("4. Comparing with current production model...")
            improvement = self.compare_models(results)
            
            if improvement > 0.02:  # 2% improvement threshold
                print(f"   ✓ Significant improvement detected: {improvement:.2%}")
                print("5. Promoting models to production...")
                self.promote_to_production(run_dir)
                self.cleanup_old_versions(keep_last=5)
                print("   ✓ Models promoted")
            else:
                print(f"   ✗ Insufficient improvement: {improvement:.2%}")
                print("   Not promoting models")
        
        except Exception as e:
            print(f"   ✗ Training failed: {e}")
            # Send alert
    
    def compare_models(self, new_results):
        """Compare new model performance with production."""
        # Load current production baseline
        baseline_file = self.model_dir / 'baseline.json'
        if baseline_file.exists():
            with open(baseline_file) as f:
                baseline = json.load(f)
        else:
            return 0  # First run
        
        # Calculate improvement
        improvement = (
            (new_results['avg_reward'] - baseline['avg_reward']) /
            baseline['avg_reward']
        )
        return improvement
    
    def promote_to_production(self, run_dir):
        """Move models from staging to production."""
        prod_link = self.model_dir / 'production_latest'
        if prod_link.exists():
            prod_link.unlink()
        prod_link.symlink_to(run_dir)
        
        # Save baseline
        baseline_file = self.model_dir / 'baseline.json'
        results = json.load((run_dir / 'evaluation_results.json').open())
        with open(baseline_file, 'w') as f:
            json.dump(results, f)
    
    def cleanup_old_versions(self, keep_last=5):
        """Remove old model versions."""
        versions = sorted(self.model_dir.glob('run_*'))
        for old_version in versions[:-keep_last]:
            import shutil
            shutil.rmtree(old_version)
            print(f"   Removed old version: {old_version.name}")
    
    def schedule_training(self):
        """Schedule periodic training jobs."""
        # Retrain daily at 2 AM
        schedule.every().day.at("02:00").do(self.retrain_models)
        
        print("Continuous training scheduled:")
        print("  - Daily retraining at 02:00")
        
        # Keep scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)

# Start background job
if __name__ == '__main__':
    manager = ContinuousTrainingManager()
    manager.schedule_training()
```

Deploy as a Kubernetes CronJob:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: markov-rl-continuous-training
  namespace: production
spec:
  # Daily at 02:00 UTC
  schedule: "0 2 * * *"
  
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: markov-rl-training
          containers:
          - name: trainer
            image: markov-rl-cache:latest
            command:
            - python
            - /app/continuous_training.py
            env:
            - name: REDIS_HOST
              value: "redis-cluster.production.svc.cluster.local"
            - name: DATA_DIR
              value: "/data/production"
            - name: MODEL_DIR
              value: "/models"
            volumeMounts:
            - name: models
              mountPath: /models
            - name: data
              mountPath: /data
            resources:
              requests:
                memory: "16Gi"
                cpu: "8"
              limits:
                memory: "32Gi"
                cpu: "16"
          
          volumes:
          - name: models
            persistentVolumeClaim:
              claimName: markov-models
          - name: data
            persistentVolumeClaim:
              claimName: training-data
          
          restartPolicy: OnFailure
  
  # Keep last 3 successful jobs
  successfulJobsHistoryLimit: 3
  # Keep last 1 failed job for debugging
  failedJobsHistoryLimit: 1
```

---

## Monitoring & Observability

### Prometheus Metrics Setup

The system exports these key metrics:

```prometheus
# Cache Performance
markov_rl_cache_hits_total{service="prod-01", endpoint="/api/users"}
markov_rl_cache_misses_total{service="prod-01", endpoint="/api/users"}
markov_rl_cache_evictions_total{service="prod-01"}
markov_rl_cache_size_bytes{service="prod-01"}

# RL Agent Performance
markov_rl_episode_reward{service="prod-01"}
markov_rl_episode_length{service="prod-01"}
markov_rl_training_loss{service="prod-01"}
markov_rl_epsilon{service="prod-01"}

# Markov Predictions
markov_rl_predictions_correct_at_k{k="1", service="prod-01"}
markov_rl_predictions_correct_at_k{k="3", service="prod-01"}
markov_rl_prediction_confidence{service="prod-01"}

# System Health
markov_rl_redis_latency_ms{service="prod-01"}
markov_rl_api_request_duration_ms{service="prod-01", endpoint="/api/users"}
markov_rl_cascade_risk_score{service="prod-01"}
```

### Grafana Dashboards

Create dashboard with these panels:

```json
{
  "dashboard": {
    "title": "Markov RL Cache Intelligence",
    "panels": [
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(markov_rl_cache_hits_total[5m]) / (rate(markov_rl_cache_hits_total[5m]) + rate(markov_rl_cache_misses_total[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Average Response Latency",
        "targets": [
          {
            "expr": "rate(markov_rl_api_request_duration_ms_sum[5m]) / rate(markov_rl_api_request_duration_ms_count[5m])"
          }
        ]
      },
      {
        "title": "Agent Learning Progress",
        "targets": [
          {
            "expr": "markov_rl_episode_reward"
          }
        ]
      },
      {
        "title": "Prediction Accuracy (Top-1)",
        "targets": [
          {
            "expr": "markov_rl_predictions_correct_at_k{k=\"1\"}"
          }
        ]
      }
    ]
  }
}
```

### Alert Rules

Create `prometheus/alert_rules.yml`:

```yaml
groups:
- name: markov_rl_cache_alerts
  rules:
  
  # Cache performance alerts
  - alert: LowCacheHitRate
    expr: |
      (rate(markov_rl_cache_hits_total[5m]) / 
       (rate(markov_rl_cache_hits_total[5m]) + 
        rate(markov_rl_cache_misses_total[5m]))) < 0.5
    for: 10m
    annotations:
      summary: "Cache hit rate below 50%"
      description: "Service {{ $labels.service }} has hit rate {{ $value | humanizePercentage }}"
  
  # High latency alerts
  - alert: HighAPILatency
    expr: |
      histogram_quantile(0.95,
        rate(markov_rl_api_request_duration_ms_bucket[5m])
      ) > 1000
    for: 5m
    annotations:
      summary: "P95 API latency above 1 second"
      description: "{{ $labels.service }} P95: {{ $value }}ms"
  
  # Cascade risk alert
  - alert: CascadeRiskHigh
    expr: markov_rl_cascade_risk_score > 0.7
    for: 5m
    annotations:
      summary: "Cascade failure risk is high"
      description: "Risk score: {{ $value | humanizePercentage }}"
  
  # Model accuracy degradation
  - alert: ModelAccuracyDegraded
    expr: |
      markov_rl_predictions_correct_at_k{k="1"} < 0.6
    for: 30m
    annotations:
      summary: "Markov prediction accuracy degraded"
      description: "Top-1 accuracy: {{ $value | humanizePercentage }}"
  
  # Service availability
  - alert: CacheIntelligenceDown
    expr: up{job="markov-rl-cache"} == 0
    for: 2m
    annotations:
      summary: "Cache Intelligence service is down"
      description: "Service {{ $labels.service }} is unreachable"
```

---

## Maintenance & Operations

### Operational Procedures

#### 1. **Daily Tasks**

```bash
#!/bin/bash
# daily_maintenance.sh

# Health check
curl -f http://localhost:8000/health || alert "Service unhealthy"

# Check cache statistics
curl http://localhost:8000/stats/cache | jq '.hit_rate'

# Monitor error rate (should be < 0.1%)
curl http://localhost:9090/api/v1/query --data-urlencode \
  'query=rate(markov_rl_api_errors_total[1h])' | jq '.data.result[0].value[1]'

# Check Redis connection
redis-cli ping

# Review logs for warnings
tail -100 /var/log/markov-cache/*.log | grep WARN
```

#### 2. **Weekly Tasks**

```bash
#!/bin/bash
# weekly_maintenance.sh

# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz /opt/models/

# Backup Redis
redis-cli BGSAVE

# Clean up old logs
find /var/log/markov-cache -mtime +30 -delete

# Performance review
# Check if model accuracy is degrading
# Review cost savings metrics
```

#### 3. **Monthly Tasks**

- Model retraining (if not automated)
- Security patches
- Dependency updates
- Capacity planning review
- Cost analysis

### Backup & Recovery

```yaml
# kubernetes/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: markov-models-backup
spec:
  schedule: "0 3 * * *"  # Daily at 3 AM
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: backup-service
          containers:
          - name: backup
            image: alpine:latest
            command:
            - /bin/sh
            - -c
            - |
              # Backup models to S3
              apk add --no-cache aws-cli
              
              BACKUP_DIR=/backup/models_$(date +\%Y\%m\%d_\%H\%M\%S)
              mkdir -p $BACKUP_DIR
              cp -r /models/* $BACKUP_DIR/
              
              # Upload to S3
              aws s3 sync $BACKUP_DIR s3://backup-bucket/markov-models/
              
              # Keep only last 30 days locally
              find /backup -mtime +30 -delete
            volumeMounts:
            - name: models
              mountPath: /models
            - name: backup-storage
              mountPath: /backup
          
          volumes:
          - name: models
            persistentVolumeClaim:
              claimName: markov-models
          - name: backup-storage
            persistentVolumeClaim:
              claimName: backup-storage
          
          restartPolicy: OnFailure
```

### Disaster Recovery

```markdown
## Disaster Recovery Plan

### Scenario 1: Model Corruption
1. Detect: Accuracy drops below threshold
2. Response:
   - Revert to last known good model version
   - Investigate cause
   - Retrain from scratch if needed
3. Time to recovery: < 1 hour

### Scenario 2: Cache Service Down
1. Detect: Health check fails for 2 minutes
2. Response:
   - Automatic restart (k8s restartPolicy)
   - Traffic routes to backup instance
   - Incident declared if persists > 5 min
3. Time to recovery: < 2 minutes

### Scenario 3: Redis Cluster Failure
1. Detect: Redis unavailable
2. Response:
   - Use in-memory cache fallback
   - Degrade to simple LRU policy
   - Alert human operators
   - Restore from backup
3. Time to recovery: 15-30 minutes

### RTO/RPO Targets:
- Recovery Time Objective: < 1 hour
- Recovery Point Objective: < 1 day
```

---

## Troubleshooting Guide

### Common Issues & Solutions

#### 1. **Low Cache Hit Rate**

```
Symptom: Hit rate < 50%

Diagnosis:
  1. Check data quality
  2. Verify Markov model training
  3. Check TTL configuration
  4. Monitor cache eviction rate

Solution:
  - Retrain Markov with more data
  - Adjust TTL (increase if evicting too fast)
  - Check cache size limits
  - Analyze access patterns
```

#### 2. **High Memory Usage**

```
Symptom: Redis memory > 90%

Solutions:
  1. Reduce cache size limit: REDIS_MAXMEMORY
  2. Adjust eviction policy
  3. Decrease TTL values
  4. Add more cache nodes
```

#### 3. **Poor Prediction Accuracy**

```
Symptom: Top-1 accuracy < 50%

Root Causes:
  1. Insufficient training data
  2. Non-stationary traffic patterns
  3. Model overfitting to old patterns
  4. New user behaviors

Solutions:
  - Increase training data window
  - Retrain more frequently
  - Use context-aware Markov
  - Validate with recent data
```

#### 4. **Agent Not Learning**

```
Symptom: Episode reward not improving

Diagnosis:
  - Check environment reward signal
  - Verify state representation
  - Check for bugs in reward calculation
  - Monitor gradient flow

Solutions:
  - Tune reward weights
  - Increase training episodes
  - Check hyperparameters
  - Debug state/reward pipeline
```

### Debugging Commands

```python
# debug_system.py
from src.integration.controller import IntegrationController
import requests

def debug_report():
    """Generate comprehensive debug report."""
    
    # 1. Service health
    print("=== Service Health ===")
    try:
        health = requests.get('http://localhost:8000/health').json()
        print(f"Status: {health['status']}")
    except Exception as e:
        print(f"✗ Service unreachable: {e}")
        return
    
    # 2. Component status
    print("\n=== Components ===")
    status = requests.get('http://localhost:8000/status').json()
    for component, is_healthy in status['component_health'].items():
        print(f"{component}: {'✓' if is_healthy else '✗'}")
    
    # 3. Metrics
    print("\n=== Metrics ===")
    metrics = requests.get('http://localhost:8001/metrics').text
    # Parse and display key metrics
    for line in metrics.split('\n'):
        if 'cache_hits' in line or 'cache_misses' in line:
            print(line)
    
    # 4. Model info
    print("\n=== Models ===")
    models = requests.get('http://localhost:8000/models/info').json()
    print(f"Markov accuracy: {models['markov_accuracy']:.2%}")
    print(f"Agent episodes trained: {models['agent_episodes']}")
    
    # 5. Recent errors
    print("\n=== Recent Errors ===")
    try:
        with open('/var/log/markov-cache/error.log') as f:
            errors = f.readlines()[-10:]
            for error in errors:
                print(error.strip())
    except FileNotFoundError:
        print("No error log found")

if __name__ == '__main__':
    debug_report()
```

---

## Migration Strategy

### For Existing Cache Solutions

If migrating from traditional caching (Redis, Memcached, etc.):

#### Phase 1: Parallel Operation (Weeks 1-2)
```
User Request
    ↓
┌─────────────────────────────────┐
│  Existing Cache (Primary)       │
│  (e.g., Redis with LRU)         │
└────────────────┬────────────────┘
                 │
                 ├─→ Also query
                 │
┌────────────────▼────────────────┐
│  Markov RL Cache (Secondary)    │
│  (Observing, learning)          │
└─────────────────────────────────┘
```

#### Phase 2: Gradual Switchover (Weeks 3-4)

```
Start: 100% old cache, 0% new
Week 3a: 90% old, 10% new (A/B testing)
Week 3b: 80% old, 20% new
Week 4a: 50% old, 50% new
Week 4b: 20% old, 80% new
End: 0% old, 100% new
```

Code for gradual switchover:

```python
# gradual_migration.py
import random

class MigrationRouter:
    def __init__(self, new_cache_percent=0):
        self.new_cache_percent = new_cache_percent
    
    def should_use_new_cache(self):
        """Determine if request should use new cache."""
        return random.random() < (self.new_cache_percent / 100.0)
    
    def get_value(self, key):
        """Get value from old or new cache."""
        if self.should_use_new_cache():
            return self.get_from_new_cache(key)
        else:
            return self.get_from_old_cache(key)
    
    def get_from_old_cache(self, key):
        """Query existing Redis/Memcached."""
        # Existing implementation
        pass
    
    def get_from_new_cache(self, key):
        """Query Markov RL cache."""
        # New implementation
        pass

# Gradually increase new_cache_percent over weeks
router = MigrationRouter(new_cache_percent=10)  # Week 1: 10%
# Later...
router = MigrationRouter(new_cache_percent=50)  # Week 3: 50%
# Eventually...
router = MigrationRouter(new_cache_percent=100)  # Week 5: 100%
```

#### Phase 3: Legacy System Decommission (Week 5+)

Once fully migrated:
1. Monitor new system for 1-2 weeks
2. Delete old cache data
3. Deallocate old infrastructure
4. Document lessons learned

---

## Security Considerations

### 1. **Access Control**

```yaml
# kubernetes/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: markov-rl-cache
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods/log"]
  verbs: ["get"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: markov-rl-cache-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: markov-rl-cache
subjects:
- kind: ServiceAccount
  name: markov-rl-cache
```

### 2. **Network Security**

```yaml
# kubernetes/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: markov-rl-cache-network
spec:
  podSelector:
    matchLabels:
      app: markov-rl-cache
  
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: production
    ports:
    - protocol: TCP
      port: 8000
  
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
```

### 3. **Secrets Management**

```bash
# Create secrets in Kubernetes
kubectl create secret generic redis-credentials \
  --from-literal=password=<strong-password> \
  -n production

kubectl create secret tls markov-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  -n production
```

### 4. **Data Privacy**

```python
# Ensure no sensitive data in logs
import logging

class SanitizedFormatter(logging.Formatter):
    def format(self, record):
        # Remove sensitive fields
        if hasattr(record, 'user_id'):
            record.user_id = '***'
        if hasattr(record, 'session_id'):
            record.session_id = '***'
        return super().format(record)
```

### 5. **Rate Limiting & DDoS Protection**

```yaml
# kubernetes/rate-limit-policy.yaml
apiVersion: networking.istio.io/v1beta1
kind: RequestAuthentication
metadata:
  name: markov-auth
spec:
  jwtRules:
  - issuer: "https://auth.example.com"
    jwksUri: "https://auth.example.com/.well-known/jwks.json"

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: markov-rl-cache-policy
spec:
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/production/sa/api-gateway"]
    to:
    - operation:
        methods: ["GET", "POST", "DELETE"]
        paths: ["/cache/*", "/api/*"]
```

---

## Final Checklist

### Pre-Go-Live

- [ ] All components deployed and healthy
- [ ] Models trained and validated
- [ ] Monitoring dashboards created
- [ ] Alert rules configured and tested
- [ ] Runbooks written and shared
- [ ] Team trained on system
- [ ] Disaster recovery procedure documented
- [ ] Security audit completed
- [ ] Performance tested and benchmarked
- [ ] SLOs defined
- [ ] Incident response plan ready

### Post-Go-Live (First 2 Weeks)

- [ ] Monitor metrics 24/7
- [ ] Respond to any alerts immediately
- [ ] Collect user feedback
- [ ] Document any issues and resolutions
- [ ] Fine-tune configurations if needed
- [ ] Prepare incident retrospective if needed
- [ ] Share success metrics with stakeholders

---

## Support & Resources

### Key Files & Locations

```
Production Deployment:
├── docker-compose.production.yml
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── rbac.yaml
├── configs/
│   └── default.yaml
├── models/
│   ├── markov_predictor.pkl
│   └── dqn_agent.pt
└── monitoring/
    ├── prometheus.yml
    ├── alert_rules.yml
    └── grafana/provisioning/
```

### Getting Help

**Documentation**:
- API Documentation: `/docs` endpoint when service is running
- Component guides: `docs/` folder
- Example code: `complete_example.py`, demos in root

**Troubleshooting**:
- Debug script: `debug_system.py`
- Health check: `GET http://localhost:8000/health`
- Logs: `/var/log/markov-cache/*.log`

---

## Conclusion

This playbook provides a complete roadmap for deploying the Markov RL API Cache system into production. The system is designed to be:

1. **Scalable**: From single-machine to multi-datacenter deployments
2. **Resilient**: Multiple redundancy and failover mechanisms
3. **Observable**: Comprehensive monitoring and logging
4. **Secure**: Industry-standard security practices
5. **Maintainable**: Clear operational procedures

Follow the phases sequentially, and your system will be production-ready in 4-6 weeks.


