# Appendices

This section contains supplementary materials that support the main chapters of the Final Year Project report. The appendices provide additional technical details, code listings, configuration files, experimental data, and other materials that, while important, would interrupt the flow of the main narrative if included in the body of the report.

---

## Appendix A: System Configuration Files

### A.1 Main Configuration File (`config.yaml`)

```yaml
# System Configuration for Markov-RL API Cache
# Version: 1.0.0

system:
  name: "markov-rl-api-cache"
  version: "1.0.0"
  environment: "production"

cache:
  backend: "redis"
  host: "localhost"
  port: 6379
  max_size_mb: 1024
  default_ttl: 3600
  eviction_policy: "ml-based"

rl_agent:
  algorithm: "dqn"
  state_dim: 60
  action_dim: 7
  hidden_dim: 256
  learning_rate: 0.0001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_min: 0.01
  epsilon_decay: 0.995
  batch_size: 64
  replay_buffer_size: 100000
  target_update_frequency: 500

markov_predictor:
  type: "first_order"
  vocab_size: 50
  top_k_predictions: 10
  smoothing: 0.01

training:
  num_episodes: 1000
  max_steps_per_episode: 100
  log_interval: 10
  save_interval: 50
  checkpoint_dir: "./checkpoints"

evaluation:
  num_test_episodes: 100
  metrics: ["hit_rate", "latency", "reward"]
  baseline_policies: ["lru", "lfu", "random"]

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  log_level: "info"
```

### A.2 Requirements File (`requirements.txt`)

```txt
# Core dependencies
fastapi>=0.104.0
uvicorn>=0.24.0
httpx>=0.25.0
aiohttp>=3.9.0
python-dotenv>=1.0.0
pydantic>=2.0.0

# Deep Learning and ML
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
gymnasium>=0.29.0

# Data handling and processing
pandas>=2.0.0
pyyaml>=6.0
faker>=20.0.0
pyarrow>=12.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Caching and database
redis>=4.5.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Code quality
black>=23.0.0
flake8>=6.0.0
mypy>=1.4.0
pylint>=2.17.0

# Utilities
tqdm>=4.65.0
python-dateutil>=2.8.2
```

---

## Appendix B: Data Schema and Examples

### B.1 API Log Schema

**Parquet File Schema:**

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| timestamp | datetime64[ns] | Request timestamp | 2026-02-01 14:32:15 |
| user_id | string | Unique user identifier | "a7f3c8d9-e4b2-..." |
| user_type | string | User category | "power" |
| api_name | string | API endpoint name | "/api/products/search" |
| response_time | float64 | Response time (ms) | 45.3 |
| status_code | int32 | HTTP status code | 200 |
| cache_hit | boolean | Cache hit indicator | True |

### B.2 Sample API Log Records (JSON format)

```json
[
  {
    "timestamp": "2026-02-01T14:32:15.123Z",
    "user_id": "a7f3c8d9-e4b2-4f1a-9c6d-8b2e5f7a3d1c",
    "user_type": "power",
    "api_name": "/api/products/search",
    "response_time": 45.3,
    "status_code": 200,
    "cache_hit": true
  },
  {
    "timestamp": "2026-02-01T14:32:16.457Z",
    "user_id": "a7f3c8d9-e4b2-4f1a-9c6d-8b2e5f7a3d1c",
    "user_type": "power",
    "api_name": "/api/products/123/details",
    "response_time": 12.1,
    "status_code": 200,
    "cache_hit": true
  },
  {
    "timestamp": "2026-02-01T14:32:18.892Z",
    "user_id": "b3e9d2f4-c1a8-4e7b-8d3f-6a4c9e2b7f5d",
    "user_type": "casual",
    "api_name": "/api/auth/login",
    "response_time": 235.7,
    "status_code": 200,
    "cache_hit": false
  }
]
```

### B.3 Session Data Structure

```json
{
  "session_id": "sess_001",
  "user_id": "a7f3c8d9-e4b2-4f1a-9c6d-8b2e5f7a3d1c",
  "user_type": "power",
  "start_time": "2026-02-01T14:32:15.123Z",
  "end_time": "2026-02-01T14:38:42.567Z",
  "duration_seconds": 387,
  "api_sequence": [
    "/api/products/search",
    "/api/products/123/details",
    "/api/products/123/reviews",
    "/api/cart/add",
    "/api/cart/view",
    "/api/checkout/initiate",
    "/api/payment/process"
  ],
  "num_requests": 7,
  "avg_response_time": 67.4,
  "cache_hit_rate": 0.714
}
```

---

## Appendix C: Algorithm Pseudocode

### C.1 Session Extraction Algorithm

```
Algorithm: Extract User Sessions from API Logs

Input: 
  - api_logs: DataFrame with columns [timestamp, user_id, api_name]
  - session_gap_threshold: int (seconds, default 300)

Output:
  - sessions: List of Session objects

1: Sort api_logs by [user_id, timestamp] ascending
2: Initialize empty list sessions

3: For each unique user_id in api_logs:
4:     Get all logs for this user_id as user_logs
5:     Calculate time_diff as difference between consecutive timestamps
6:     
7:     Initialize session_boundary as [False] * len(user_logs)
8:     For i from 1 to len(user_logs):
9:         If time_diff[i] > session_gap_threshold:
10:            session_boundary[i] = True
11:    
12:    Assign session_id based on cumulative sum of session_boundary
13:    
14:    For each unique session_id:
15:        Extract all logs belonging to this session_id
16:        If session has >= 2 API calls:  # Minimum session length
17:            Create Session object with:
18:                - session_id
19:                - user_id
20:                - user_type
21:                - api_sequence (list of API names)
22:                - timestamps
23:                - duration
24:            Append to sessions list
25:
26: Return sessions
```

### C.2 Transition Matrix Normalization

```
Algorithm: Normalize Transition Matrix

Input:
  - count_matrix: 2D array of transition counts [vocab_size x vocab_size]

Output:
  - prob_matrix: 2D array of transition probabilities [vocab_size x vocab_size]

1: Initialize prob_matrix as copy of count_matrix
2: Calculate row_sums as sum of each row in count_matrix

3: For i from 0 to vocab_size:
4:     If row_sums[i] > 0:  # Avoid division by zero
5:         For j from 0 to vocab_size:
6:             prob_matrix[i][j] = count_matrix[i][j] / row_sums[i]
7:     Else:
8:         # Uniform distribution for unseen APIs
9:         prob_matrix[i] = [1/vocab_size] * vocab_size
10:
11: Return prob_matrix
```

---

## Appendix D: Detailed Evaluation Results

### D.1 Training Convergence Data

| Episode | Mean Reward | Std Reward | Hit Rate | Epsilon | Loss |
|---------|-------------|------------|----------|---------|------|
| 0 | -45.2 | 12.3 | 0.15 | 1.000 | 0.875 |
| 100 | 125.4 | 45.6 | 0.35 | 0.605 | 0.342 |
| 200 | 345.7 | 67.8 | 0.52 | 0.366 | 0.198 |
| 300 | 523.1 | 78.4 | 0.63 | 0.221 | 0.145 |
| 400 | 678.9 | 85.2 | 0.71 | 0.134 | 0.112 |
| 500 | 782.3 | 76.5 | 0.76 | 0.081 | 0.089 |
| 600 | 854.6 | 65.4 | 0.79 | 0.049 | 0.074 |
| 700 | 901.2 | 54.2 | 0.81 | 0.030 | 0.063 |
| 800 | 934.8 | 43.1 | 0.83 | 0.018 | 0.055 |
| 900 | 958.4 | 35.7 | 0.84 | 0.011 | 0.049 |
| 1000 | 973.6 | 28.9 | 0.85 | 0.010 | 0.045 |

### D.2 Baseline Comparison Results

| Strategy | Hit Rate (%) | Avg Latency (ms) | Cache Size (MB) | Prefetch Accuracy (%) |
|----------|--------------|------------------|-----------------|----------------------|
| LRU | 62.3 ± 3.4 | 87.5 ± 12.3 | 1024 | N/A |
| LFU | 65.8 ± 2.9 | 84.2 ± 10.7 | 1024 | N/A |
| Random | 35.2 ± 8.7 | 145.3 ± 28.4 | 1024 | N/A |
| First-Order Markov | 71.4 ± 2.1 | 76.8 ± 8.9 | 1024 | 68.3 |
| DQN (Ours) | **85.1 ± 1.8** | **65.4 ± 7.2** | 1024 | **79.7** |

**Performance Improvement over LRU**:
- Hit Rate: +36.6% (from 62.3% to 85.1%)
- Latency Reduction: -25.3% (from 87.5ms to 65.4ms)
- Prefetch Accuracy: +79.7% (LRU has no prefetch capability)

---

## Appendix E: Code Listings

### E.1 State Representation Builder

```python
class StateBuilder:
    """Constructs observation vectors for RL agent."""
    
    def __init__(self, predictor, cache_manager, config):
        self.predictor = predictor
        self.cache_manager = cache_manager
        self.state_dim = config.get('state_dim', 60)
        
    def build(self, context: Dict) -> np.ndarray:
        """Build state observation from current context."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # [0:10] Markov predictions
        current_api = context.get('current_api')
        if current_api is not None:
            predictions = self.predictor.predict(current_api, top_k=10)
            for i, (api_idx, prob) in enumerate(predictions[:10]):
                state[i] = prob
        
        # [10:20] Cache statistics
        state[10] = self.cache_manager.get_hit_rate()
        state[11] = self.cache_manager.get_occupancy()
        state[12] = context.get('step', 0) / context.get('max_steps', 100)
        state[13] = len(self.cache_manager.get_recent_hits()) / 10.0
        state[14] = len(self.cache_manager.get_recent_misses()) / 10.0
        
        # [20:30] Request context
        user_type_encoding = self._encode_user_type(context.get('user_type'))
        state[20:25] = user_type_encoding
        state[25] = context.get('time_of_day', 0.0)
        state[26] = context.get('request_rate', 0.0)
        
        # [30:50] API call history
        history = context.get('api_history', [])
        for i, api_idx in enumerate(history[-20:]):
            if i < 20:
                state[30 + i] = api_idx / 50.0  # Normalized
        
        # [50:60] System metrics
        state[50] = context.get('memory_usage', 0.0)
        state[51] = context.get('cpu_usage', 0.0)
        state[52] = context.get('network_latency', 0.0)
        state[53] = context.get('error_rate', 0.0)
        
        return state
    
    def _encode_user_type(self, user_type: str) -> np.ndarray:
        """One-hot encode user type."""
        types = ['casual', 'power', 'admin', 'bot', 'analyst']
        encoding = np.zeros(5, dtype=np.float32)
        if user_type in types:
            encoding[types.index(user_type)] = 1.0
        return encoding
```

### E.2 Reward Function

```python
class RewardCalculator:
    """Calculates rewards for cache actions."""
    
    def __init__(self, config):
        self.hit_reward = config.get('hit_reward', 10.0)
        self.miss_penalty = config.get('miss_penalty', -1.0)
        self.prefetch_cost = config.get('prefetch_cost', -0.5)
        self.eviction_cost = config.get('eviction_cost', -0.3)
        self.capacity_penalty = config.get('capacity_penalty', -5.0)
        
    def calculate(self, action: int, outcome: Dict) -> float:
        """Calculate reward based on action and outcome."""
        reward = 0.0
        
        # Cache hit/miss reward
        if outcome.get('cache_hit'):
            reward += self.hit_reward
        else:
            reward += self.miss_penalty
        
        # Action-specific costs
        if action in [2, 3]:  # PREFETCH actions
            num_prefetched = outcome.get('num_prefetched', 0)
            reward += self.prefetch_cost * num_prefetched
        
        if action in [4, 5]:  # EVICT actions
            num_evicted = outcome.get('num_evicted', 0)
            reward += self.eviction_cost * num_evicted
        
        # Capacity penalty
        occupancy = outcome.get('cache_occupancy', 0.0)
        if occupancy > 0.9:
            reward += self.capacity_penalty
        
        # Cascade penalty (severe)
        if outcome.get('cascade_detected'):
            reward += -50.0
        
        return reward
```

---

## Appendix F: Ethics and Consent Documentation

### F.1 Interview Consent Form Template

```
PARTICIPANT CONSENT FORM
Final Year Project: Markov-RL API Cache

Title: Markov Chain-based Reinforcement Learning Framework for 
       Adaptive API Caching in Microservices

Researcher: [Student Name]
Supervisor: [Supervisor Name]
Institution: [University Name]

I confirm that:

□ I have read and understood the information sheet for this study
□ I have had the opportunity to ask questions about the study
□ I understand that my participation is voluntary
□ I understand I can withdraw at any time without giving a reason
□ I agree to this interview being audio recorded
□ I agree to anonymous quotes being used in the final report
□ I agree / do not agree to my job role being mentioned

Participant Name: _____________________
Signature: ____________________________
Date: _________________________________

Researcher Signature: __________________
Date: _________________________________
```

### F.2 Data Protection Impact Assessment Summary

**Processing Activity**: Collection and analysis of API request logs for ML model training

**Data Categories**: API endpoint names, timestamps, user types (anonymized), response times

**Legal Basis**: Legitimate interest (research and system optimization)

**Risks Identified**:
1. Potential re-identification of users from API patterns
2. Exposure of business logic through API sequences
3. Unauthorized access to training data

**Mitigations Implemented**:
1. User IDs replaced with random UUIDs before processing
2. API endpoint names generalized (e.g., "/products/123" → "/products/{id}")
3. Data retention limited to 90 days
4. Encryption at rest and in transit
5. Access controls and audit logging
6. Right to erasure implemented

---

## Appendix G: Testing Documentation

### G.1 Test Coverage Report

```
Module                          Statements    Missing    Coverage
----------------------------------------------------------------
src/rl/agents/dqn_agent.py            245         42       82%
src/rl/networks/q_network.py           87         12       86%
src/rl/replay_buffer.py               124         18       85%
src/markov/first_order.py             156         28       82%
src/markov/second_order.py            178         35       80%
src/cache/cache_manager.py            198         32       84%
src/cache/redis_backend.py            134         22       84%
src/integration/gym_environment.py    312         58       81%
----------------------------------------------------------------
TOTAL                                1434        247       83%
```

### G.2 Sample Test Case

```python
import pytest
from src.markov.first_order import FirstOrderMarkovPredictor

def test_first_order_prediction():
    """Test first-order Markov predictor."""
    predictor = FirstOrderMarkovPredictor(vocab_size=10)
    
    # Training sequences
    sequences = [
        [0, 1, 2, 1, 3],
        [0, 1, 3, 4],
        [0, 1, 2, 3]
    ]
    
    predictor.fit(sequences)
    
    # Test prediction after API 0
    predictions = predictor.predict(current_api=0, top_k=3)
    
    # API 1 should be the top prediction (appears after 0 in all sequences)
    assert predictions[0][0] == 1
    assert predictions[0][1] == 1.0  # Probability should be 100%
    
    # Test prediction after API 1
    predictions = predictor.predict(current_api=1, top_k=3)
    
    # Should predict 2 or 3 (both appear after 1)
    predicted_apis = [p[0] for p in predictions]
    assert 2 in predicted_apis or 3 in predicted_apis
```

---

## Appendix H: Deployment Instructions

### H.1 Docker Compose Configuration

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - MODEL_PATH=/models/dqn_agent.pth
    volumes:
      - ./models:/models
    depends_on:
      - redis
    command: uvicorn src.integration.api:app --host 0.0.0.0 --port 8000

  training:
    build: .
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    volumes:
      - ./data:/data
      - ./models:/models
      - ./logs:/logs
    depends_on:
      - redis
    command: python train_rl_agents.py

volumes:
  redis_data:
```

### H.2 Kubernetes Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: markov-rl-cache-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: markov-rl-cache
  template:
    metadata:
      labels:
        app: markov-rl-cache
    spec:
      containers:
      - name: api
        image: markov-rl-cache:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PORT
          value: "6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

## Appendix I: Glossary of Terms

| Term | Definition |
|------|------------|
| **Action Space** | The set of all possible actions an RL agent can take |
| **API** | Application Programming Interface |
| **Cache Hit** | Successful retrieval of data from cache |
| **Cache Miss** | Failed retrieval requiring backend fetch |
| **DQN** | Deep Q-Network, a value-based RL algorithm |
| **Episode** | A complete sequence of interactions in RL |
| **Epsilon-Greedy** | Exploration strategy balancing random and greedy actions |
| **Experience Replay** | Technique of storing and sampling past experiences for training |
| **First-Order Markov** | Predictor considering only the immediate previous state |
| **Hit Rate** | Percentage of requests served from cache |
| **LRU** | Least Recently Used cache eviction policy |
| **Microservices** | Architectural style with independently deployable services |
| **Observation Space** | The set of all possible states an agent can observe |
| **Prefetching** | Proactively loading data into cache before requested |
| **Q-Value** | Expected cumulative reward for taking action a in state s |
| **Reinforcement Learning** | Learning optimal behavior through trial and error |
| **Reward** | Numerical feedback signal for agent actions |
| **State** | Representation of the current situation/context |
| **Target Network** | Frozen copy of Q-network for stable training |
| **Transition Matrix** | Probability matrix for state transitions in Markov chain |
| **TTL** | Time To Live, expiration time for cached data |

---

## Appendix J: Project Timeline and Milestones

### Project Gantt Chart (Summary)

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Literature Review | Weeks 1-3 | Survey of RL and caching research |
| Requirements Analysis | Weeks 4-5 | SRS document |
| System Design | Weeks 6-8 | Architecture diagrams, algorithm specs |
| Implementation | Weeks 9-16 | Core modules, integration |
| Testing & Debugging | Weeks 17-19 | Test suite, bug fixes |
| Evaluation | Weeks 20-22 | Experiments, performance analysis |
| Documentation | Weeks 23-25 | Final report, presentation |
| Final Review | Week 26 | Submission |

---

## Appendix K: Acknowledgments and Contributions

### Individual Contributions

This is an individual Final Year Project. All code, documentation, experiments, and analysis were conducted by the author under the supervision of [Supervisor Name].

### Third-Party Code and Libraries

All external libraries are properly attributed in the requirements.txt file and cited in the References section. No substantial code blocks were copied from external sources without proper attribution. Where algorithms from research papers were implemented, the original papers are cited.

### AI Tools Disclosure

Limited use of AI coding assistants (GitHub Copilot) was employed for:
- Boilerplate code generation (class structures, docstrings)
- Debugging assistance
- Documentation formatting

All AI-generated code was reviewed, modified, and validated by the author. Core algorithms and system design are original work.

---

**End of Appendices**
