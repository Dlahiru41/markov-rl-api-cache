# Prometheus + Grafana Monitoring — Markov-RL API Cache

This document describes the full observability stack implemented for the
**Markov-RL Intelligent API Caching** system.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Markov-RL Cache System                                              │
│                                                                      │
│  ┌──────────────────┐    scrape :9200    ┌──────────────────────┐   │
│  │  MetricsCollector │◄──────────────────│  Prometheus :9090    │   │
│  │  (prometheus_     │                   │                      │   │
│  │   client HTTP)    │                   │  alert_rules.yml     │   │
│  └──────────┬────────┘                   └──────────┬───────────┘   │
│             │ records                               │ datasource    │
│  ┌──────────▼────────────────────────┐   ┌─────────▼───────────┐   │
│  │  IntegrationController             │   │  Grafana :3000       │   │
│  │  ├─ DQN Agent (training)           │   │                      │   │
│  │  ├─ MarkovPredictor                │   │  markov_rl_cache     │   │
│  │  ├─ CacheManager (Redis)           │   │  dashboard           │   │
│  │  └─ CachingEnv (Gym)              │   └─────────────────────┘   │
│  └───────────────────────────────────┘                              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  FastAPI :8080   GET /metrics  →  Prometheus exposition      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Start the full stack with Docker Compose

```bash
cd docker/
docker-compose up -d
```

Services started:
| Service | URL | Purpose |
|---|---|---|
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| Prometheus | http://localhost:9090 | Metrics storage |
| RL Cache Exporter | http://localhost:9200/metrics | Prometheus scrape target |
| Redis Exporter | http://localhost:9121/metrics | Redis metrics |
| REST API | http://localhost:8080 | Control API + /metrics |

### 2. Run standalone (development)

```bash
# Install dependencies
pip install prometheus-client

# Start metrics exporter + training
python -m src.monitoring.exporter --port 9200 --api-port 8080

# Or enable monitoring inside your own controller
from src.integration.controller import IntegrationController, ControllerConfig
config = ControllerConfig(mode="training", enable_monitoring=True)
controller = IntegrationController(config)
controller.setup()
controller.start()
controller.train()
```

### 3. Use MetricsCollector directly

```python
from src.monitoring import MetricsCollector, start_metrics_server

# Create collector
collector = MetricsCollector(service="my-api-gateway")

# Start HTTP server (Prometheus scrapes :9200/metrics)
start_metrics_server(port=9200, registry=collector.registry)

# Record events
collector.record_cache_hit(endpoint="/api/products")
collector.record_cache_miss(endpoint="/api/checkout")
collector.record_episode(reward=342.5, length=85, hit_rate=0.77, cascade_occurred=False)
collector.record_training_step(loss=0.042, epsilon=0.14)
collector.update_cascade_risk(0.22)
collector.record_markov_prediction(
    correct_at_k={1: True, 3: True, 5: False},
    confidence=0.82,
    vocab_size=25
)
```

---

## Metrics Reference

All metrics are prefixed with `markov_rl_` and labelled with `service`.

### Cache Performance

| Metric | Type | Description |
|--------|------|-------------|
| `markov_rl_cache_hit_rate` | Gauge | Rolling hit rate [0,1] |
| `markov_rl_cache_hits_total` | Counter | Total hits (by endpoint) |
| `markov_rl_cache_misses_total` | Counter | Total misses (by endpoint) |
| `markov_rl_cache_entries` | Gauge | Current number of cached entries |
| `markov_rl_cache_utilization` | Gauge | Fraction of capacity used [0,1] |
| `markov_rl_cache_evictions_total` | Counter | Evictions (by strategy: lru\|low_prob\|ttl) |
| `markov_rl_cache_sets_total` | Counter | Set (write) operations |
| `markov_rl_cache_entry_size_bytes` | Histogram | Individual entry sizes |
| `markov_rl_cache_operation_latency_seconds` | Histogram | Backend get/set latency |

### Prefetch Engine

| Metric | Type | Description |
|--------|------|-------------|
| `markov_rl_prefetch_requests_total` | Counter | Prefetch attempts (by strategy) |
| `markov_rl_prefetch_hits_total` | Counter | Prefetched items that were used |
| `markov_rl_prefetch_wasted_total` | Counter | Prefetched items that expired unused |
| `markov_rl_prefetch_efficiency` | Gauge | hits/requests ratio [0,1] |
| `markov_rl_prefetch_bandwidth_bytes_total` | Counter | Bytes transferred for prefetching |

### Markov Chain Predictor

| Metric | Type | Description |
|--------|------|-------------|
| `markov_rl_markov_predictions_total` | Counter | Predictions made (by order: 1\|2) |
| `markov_rl_markov_correct_total` | Counter | Correct predictions (by k) |
| `markov_rl_markov_accuracy_topk` | Gauge | Rolling accuracy (by k: 1\|3\|5) |
| `markov_rl_markov_confidence` | Histogram | Top-1 prediction confidence |
| `markov_rl_markov_vocab_size` | Gauge | Known API endpoints |
| `markov_rl_markov_transition_entropy` | Gauge | Shannon entropy of transition dist. |

### Reinforcement Learning Agent

| Metric | Type | Description |
|--------|------|-------------|
| `markov_rl_rl_episodes_total` | Counter | Completed training episodes |
| `markov_rl_rl_steps_total` | Counter | Total environment steps |
| `markov_rl_rl_episode_reward` | Histogram | Reward per episode |
| `markov_rl_rl_episode_reward_mean` | Gauge | Rolling mean reward (last 100 eps) |
| `markov_rl_rl_episode_length` | Histogram | Steps per episode |
| `markov_rl_rl_epsilon` | Gauge | Exploration rate ε [0,1] |
| `markov_rl_rl_loss` | Gauge | Latest TD-error / training loss |
| `markov_rl_rl_q_value_mean` | Gauge | Mean Q-value (diagnostics) |
| `markov_rl_rl_replay_buffer_size` | Gauge | Transitions in replay buffer |
| `markov_rl_rl_target_updates_total` | Counter | Target network hard-updates |
| `markov_rl_rl_action_counts_total` | Counter | Actions selected (by action name) |
| `markov_rl_rl_training_steps_total` | Counter | Gradient update steps |

### Cascade Prevention

| Metric | Type | Description |
|--------|------|-------------|
| `markov_rl_cascade_risk_score` | Gauge | Current cascade risk [0,1] |
| `markov_rl_cascade_events_total` | Counter | Cascade failures detected |
| `markov_rl_cascade_prevented_total` | Counter | Cascades prevented by RL agent |
| `markov_rl_cascade_prevention_rate` | Gauge | prevented/(prevented+occurred) [0,1] |

### Reward Decomposition

| Metric | Type | Description |
|--------|------|-------------|
| `markov_rl_reward_component` | Gauge | Per-component reward (cache\|cascade\|prefetch\|latency\|bandwidth\|shaping) |

### System / Infrastructure

| Metric | Type | Description |
|--------|------|-------------|
| `markov_rl_request_latency_seconds` | Histogram | End-to-end API latency (p50/p95/p99) |
| `markov_rl_request_count_total` | Counter | Requests served (by status code) |
| `markov_rl_backend_call_latency_seconds` | Histogram | Cache-miss backend call latency |
| `markov_rl_system_cpu_usage` | Gauge | CPU fraction [0,1] |
| `markov_rl_system_memory_usage` | Gauge | Memory fraction [0,1] |
| `markov_rl_active_sessions` | Gauge | Concurrent user sessions |
| `markov_rl_session_length` | Histogram | API calls per user session |
| `markov_rl_requests_per_second` | Gauge | Rolling RPS |

---

## Grafana Dashboard

The dashboard `docker/monitoring/dashboards/markov_rl_cache.json` is
**auto-provisioned** when Grafana starts. It contains **8 sections**
with **35+ panels**:

| Section | Panels | What it shows |
|---------|--------|---------------|
| 🎯 Executive Summary | 6 stat cards | Hit rate, cascade risk, mean reward, Markov accuracy, prevention rate, utilisation |
| 📦 Cache Performance | 6 panels | Hit/miss rates, entry count, utilisation, evictions, op latency, entry sizes |
| ⚡ Prefetch Engine | 4 panels | Efficiency, rate by strategy, bandwidth, efficiency gauge |
| 🔗 Markov Predictor | 4 panels | Top-k accuracy, prediction rate + vocab, transition entropy, confidence heatmap |
| 🤖 RL Agent | 7 panels | Reward curve, epsilon decay, loss+Q-value, replay buffer, action distribution, throughput, reward decomposition |
| 🛡️ Cascade Prevention | 5 panels | Risk score, prevented vs occurred, prevention gauge, total events counters |
| 🌐 API Latency | 4 panels | p50/p95/p99 latency, RPS + sessions, backend latency, status code rates |
| 🖥️ System Resources | 3 panels | CPU, memory, object counts |

**Dashboard UID:** `markov-rl-cache-v1`  
**Auto-refresh:** 30 seconds  
**Default range:** last 1 hour

---

## Alert Rules

Defined in `docker/monitoring/alert_rules.yml` — **16 alerts** across 5 groups:

### Cache Performance
| Alert | Condition | Severity |
|-------|-----------|----------|
| `CacheHitRateLow` | hit_rate < 50% for 5m | warning |
| `CacheHitRateCritical` | hit_rate < 30% for 3m | critical |
| `CacheUtilisationHigh` | utilisation > 90% for 5m | warning |
| `CacheEvictionRateHigh` | evictions > 50/min for 3m | warning |

### Cascade Prevention
| Alert | Condition | Severity |
|-------|-----------|----------|
| `CascadeRiskElevated` | risk_score > 0.60 for 1m | warning |
| `CascadeRiskCritical` | risk_score > 0.85 for 30s | critical |
| `CascadeEventDetected` | any cascade in 5m | critical |
| `CascadePreventionRateDropped` | prevention_rate < 80% for 10m | warning |

### RL Agent Health
| Alert | Condition | Severity |
|-------|-----------|----------|
| `RLTrainingLossSpike` | loss > 10.0 for 5m | warning |
| `RLEpisodeRewardDeclining` | reward derivative < -1/min for 15m | warning |
| `RLExplorationStuck` | epsilon > 0.50 after 500+ episodes | warning |
| `RLReplayBufferLow` | buffer < 500 after 10+ episodes | warning |
| `RLNoTrainingProgress` | no gradient steps in 10m | warning |

### Markov Predictor
| Alert | Condition | Severity |
|-------|-----------|----------|
| `MarkovAccuracyLow` | top-1 accuracy < 25% after 100+ predictions | warning |
| `MarkovPredictionStopped` | no predictions in 5m | critical |

### Infrastructure
| Alert | Condition | Severity |
|-------|-----------|----------|
| `APILatencyHigh` | p99 latency > 200ms for 5m | warning |
| `BackendLatencyHigh` | p95 backend latency > 500ms for 5m | warning |
| `HighCPUUsage` | cpu > 85% for 5m | warning |
| `HighMemoryUsage` | memory > 90% for 5m | critical |
| `PrefetchEfficiencyLow` | efficiency < 20% when active for 10m | warning |

---

## Files Created

```
src/monitoring/
├── __init__.py          # Package exports
├── metrics.py           # MetricsCollector — 40+ metrics, all Prometheus types
└── exporter.py          # Standalone exporter + controller integration helpers

docker/monitoring/
├── prometheus.yml       # Scrape config (RL exporter, Redis, microservices)
├── alert_rules.yml      # 20 alert rules across 5 groups
├── datasources/
│   └── prometheus.yml   # Grafana datasource auto-provisioning
└── dashboards/
    ├── dashboard.yml    # Grafana dashboard auto-provisioning config
    └── markov_rl_cache.json  # 8-section, 35+ panel Grafana dashboard
```

---

## Design Decisions

### Why these specific metrics?

1. **Cache hit rate** is the primary KPI — directly maps to latency reduction and infrastructure cost savings.

2. **Cascade risk score** is the most critical safety metric — a single cascade event equals -100 in the reward function (10× worse than a cache miss), so early warning is essential.

3. **Reward decomposition** (cache + cascade + prefetch + latency + bandwidth + shaping) lets engineers understand *why* the agent is behaving a certain way — essential for debugging divergence.

4. **Epsilon decay** tracks the exploration→exploitation transition; stagnation here reveals hyperparameter issues.

5. **Markov top-k accuracy** at k=1,3,5 maps directly to prefetch strategy viability (conservative=k1, moderate=k3, aggressive=k5).

6. **Prefetch efficiency** (used/requested) prevents the agent from over-prefetching — waste above 80% means bandwidth cost exceeds benefit.

7. **Replay buffer size** guards against training with too little experience, which causes overfit to recent transitions.

8. **Transition entropy** of the Markov chain signals how predictable the traffic is — high entropy means the RL agent must rely more on its own policy than on Markov hints.

