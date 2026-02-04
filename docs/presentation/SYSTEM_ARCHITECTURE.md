# System Architecture Documentation
## Markov-RL API Cache System

**Document Version:** 1.0  
**Date:** February 2026  
**Status:** Complete

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Architecture](#component-architecture)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Deployment Architecture](#deployment-architecture)
6. [User Interface Components](#user-interface-components)
7. [Integration Points](#integration-points)
8. [Security Architecture](#security-architecture)
9. [Scalability Architecture](#scalability-architecture)

---

## 1. Executive Summary

This document presents the comprehensive system architecture for the Markov-RL API Cache System. The architecture follows a modular, layered design that separates concerns and enables independent development, testing, and deployment of components.

**Key Architectural Principles:**
- **Modularity:** Clear separation between components
- **Extensibility:** Easy to add new features and algorithms
- **Testability:** Each component independently testable
- **Production-Ready:** Built for real-world deployment
- **Standards-Based:** Uses industry-standard interfaces (Gymnasium, Redis)

---

## 2. High-Level Architecture

### 2.1 System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        EXTERNAL LAYER                            │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Client     │  │     API      │  │   Monitoring │          │
│  │ Applications │  │   Gateway    │  │    Tools     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                  │                  │                  │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTELLIGENT CACHE LAYER                       │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Cache Controller & Manager                    │  │
│  │  • Request routing     • Cache operations                 │  │
│  │  • Response handling   • Metrics tracking                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         │                    │                     │             │
│         ▼                    ▼                     ▼             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐        │
│  │   Markov    │    │  DQN Agent   │    │   Cache     │        │
│  │  Predictor  │───▶│  (RL Brain)  │───▶│  Backend    │        │
│  │             │    │              │    │             │        │
│  │ • Patterns  │    │ • Actions    │    │ • Memory    │        │
│  │ • Predict   │    │ • Learning   │    │ • Redis     │        │
│  └─────────────┘    └──────────────┘    └─────────────┘        │
│         │                    │                     │             │
└─────────┼────────────────────┼─────────────────────┼─────────────┘
          │                    │                     │
          ▼                    ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   REINFORCEMENT LEARNING LAYER                   │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                Gymnasium Environment                       │  │
│  │  • Episode management    • Reward calculation             │  │
│  │  • State construction    • Metrics tracking               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BACKEND SERVICES LAYER                      │
│                                                                   │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐   │
│  │ Auth │  │ User │  │Product│  │ Cart │  │Order │  │Payment│   │
│  │Service│  │Service│  │Service│  │Service│  │Service│  │Service│   │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Responsibilities

**External Layer:**
- Client applications make API requests
- API Gateway routes requests
- Monitoring tools observe system behavior

**Intelligent Cache Layer:**
- Intercepts and processes all API requests
- Makes caching decisions using RL agent
- Manages cache operations and storage

**Reinforcement Learning Layer:**
- Provides training environment for RL agents
- Manages episodes and learning process
- Evaluates policy performance

**Backend Services Layer:**
- Microservices providing business logic
- Source of data for cache
- Target for cache miss requests

---

## 3. Component Architecture

### 3.1 Component Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                         src/markov/                                │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────────────────────────────────────────┐       │
│  │  MarkovPredictor (Facade)                              │       │
│  │  ───────────────────────────────────────────────────   │       │
│  │  + fit(sequences, contexts)                            │       │
│  │  + predict(current_state, k, context) → predictions    │       │
│  │  + observe(api_call, context)                          │       │
│  │  + reset_history()                                     │       │
│  └────────────────────────────────────────────────────────┘       │
│           │                  │                  │                  │
│           ▼                  ▼                  ▼                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ FirstOrder   │  │ SecondOrder  │  │ContextAware  │           │
│  │ Predictor    │  │ Predictor    │  │ Predictor    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│           │                  │                  │                  │
│           └──────────────────┴──────────────────┘                  │
│                              │                                     │
│                              ▼                                     │
│                   ┌──────────────────┐                            │
│                   │ TransitionMatrix │                            │
│                   │ • P(next|current)│                            │
│                   └──────────────────┘                            │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                           src/rl/                                  │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────────────────────────────────────────┐       │
│  │  DQNAgent                                              │       │
│  │  ────────────────────────────────────────────────────  │       │
│  │  + select_action(state) → action                       │       │
│  │  + store_transition(s, a, r, s', done)                 │       │
│  │  + train_step() → loss                                 │       │
│  │  + save_model(path)                                    │       │
│  │  + load_model(path)                                    │       │
│  └────────────────────────────────────────────────────────┘       │
│           │                                  │                     │
│           ▼                                  ▼                     │
│  ┌──────────────┐                  ┌──────────────┐              │
│  │  QNetwork    │                  │ReplayBuffer  │              │
│  │  (Online)    │                  │• capacity    │              │
│  ├──────────────┤                  │• sample()    │              │
│  │ [256,256,128]│                  └──────────────┘              │
│  └──────────────┘                                                 │
│           │                                                        │
│           ▼                                                        │
│  ┌──────────────┐                                                 │
│  │  QNetwork    │                                                 │
│  │  (Target)    │                                                 │
│  └──────────────┘                                                 │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ StateBuilder │  │ActionSpace   │  │RewardCalc    │           │
│  │ • 60-dim vec │  │ • 7 actions  │  │ • Multi-obj  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                          src/cache/                                │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────────────────────────────────────────┐       │
│  │  CacheManager (Facade)                                 │       │
│  │  ───────────────────────────────────────────────────   │       │
│  │  + get(key) → value                                    │       │
│  │  + set(key, value, ttl)                                │       │
│  │  + delete(key)                                         │       │
│  │  + evict_items(strategy)                               │       │
│  │  + prefetch(api_list)                                  │       │
│  │  + get_metrics() → dict                                │       │
│  └────────────────────────────────────────────────────────┘       │
│           │                                  │                     │
│           ▼                                  ▼                     │
│  ┌──────────────┐                  ┌──────────────┐              │
│  │ InMemory     │                  │ Redis        │              │
│  │ Backend      │                  │ Backend      │              │
│  │ • Fast       │                  │ • Distributed│              │
│  │ • Dev/Test   │                  │ • Production │              │
│  └──────────────┘                  └──────────────┘              │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Serializer   │  │ Compressor   │  │ TTLManager   │           │
│  │ • Pickle     │  │ • zlib       │  │ • Expiry     │           │
│  │ • JSON       │  │ • Auto       │  │ • Cleanup    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└───────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                      src/integration/                              │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────────────────────────────────────────┐       │
│  │  CachingEnv (gymnasium.Env)                            │       │
│  │  ───────────────────────────────────────────────────   │       │
│  │  + reset() → (observation, info)                       │       │
│  │  + step(action) → (obs, reward, term, trunc, info)     │       │
│  │  + render()                                            │       │
│  │  + close()                                             │       │
│  │                                                        │       │
│  │  observation_space: Box(60,)                           │       │
│  │  action_space: Discrete(7)                             │       │
│  └────────────────────────────────────────────────────────┘       │
│                              │                                     │
│           ┌──────────────────┼──────────────────┐                 │
│           ▼                  ▼                  ▼                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ Controller   │  │  Simulator   │  │  Metrics     │           │
│  │ • Routes     │  │ • Sessions   │  │ • Tracking   │           │
│  │ • Handles    │  │ • Workloads  │  │ • Reporting  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└───────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Descriptions

**Markov Components:**

1. **MarkovPredictor (Facade)**
   - Main interface for pattern prediction
   - Delegates to specific predictor implementations
   - Manages predictor lifecycle and configuration

2. **FirstOrderPredictor**
   - Learns P(next_api | current_api)
   - Simple, fast, good baseline
   - Implementation: `src/markov/first_order.py`

3. **SecondOrderPredictor**
   - Learns P(next_api | prev_api, current_api)
   - Captures longer-term patterns
   - Implementation: `src/markov/second_order.py`

4. **ContextAwarePredictor**
   - Learns P(next_api | context, current_api)
   - Considers user type, time, system state
   - Implementation: `src/markov/context_aware.py`

5. **TransitionMatrix**
   - Stores and updates probability distributions
   - Efficient sparse matrix representation
   - Supports smoothing techniques

**RL Components:**

1. **DQNAgent**
   - Main reinforcement learning agent
   - Implements Deep Q-Learning algorithm
   - Manages training and inference
   - Implementation: `src/rl/agents/dqn_agent.py`

2. **QNetwork (Online & Target)**
   - Neural network for Q-value approximation
   - Architecture: Input(60) → [256, 256, 128] → Output(7)
   - ReLU activations, Adam optimizer
   - Implementation: `src/rl/networks/q_network.py`

3. **ReplayBuffer**
   - Stores experience tuples (s, a, r, s', done)
   - Capacity: 100,000 transitions
   - Uniform random sampling
   - Implementation: `src/rl/replay_buffer.py`

4. **StateBuilder**
   - Constructs 60-dimensional state vectors
   - Normalizes all features to [0, 1]
   - Implementation: `src/rl/state.py`

5. **ActionSpace**
   - Defines 7 discrete cache actions
   - Encodes/decodes action semantics
   - Implementation: `src/rl/actions.py`

6. **RewardCalculator**
   - Multi-objective reward function
   - Balances cache efficiency, latency, cascades
   - Implementation: `src/rl/reward.py`

**Cache Components:**

1. **CacheManager (Facade)**
   - High-level cache operations
   - Backend-agnostic interface
   - Handles serialization, compression, TTL
   - Implementation: `src/cache/cache_manager.py`

2. **InMemoryBackend**
   - Fast local cache storage
   - Used for development and testing
   - Implementation: `src/cache/backend.py`

3. **RedisBackend**
   - Distributed cache storage
   - Used for production deployment
   - Implementation: `src/cache/redis_backend.py`

4. **Serializer**
   - Converts objects to bytes (pickle/JSON)
   - Handles deserialization

5. **Compressor**
   - zlib compression for large objects
   - Automatic compression above threshold

6. **TTLManager**
   - Enforces time-to-live for cached items
   - Background cleanup of expired items

**Integration Components:**

1. **CachingEnv**
   - OpenAI Gymnasium environment
   - Orchestrates all components
   - Manages training episodes
   - Implementation: `src/integration/gym_environment.py`

2. **Controller**
   - Routes API requests
   - Coordinates caching decisions
   - Implementation: `src/integration/controller.py`

3. **Simulator**
   - Generates realistic API traffic
   - Multiple workload types
   - Implementation: `simulator/` directory

4. **MetricsCollector**
   - Tracks performance metrics
   - Aggregates statistics
   - Exports to monitoring systems

---

## 4. Data Flow Architecture

### 4.1 Request Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     INCOMING API REQUEST                         │
│                    /api/products/123                             │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: Cache Lookup                                            │
├─────────────────────────────────────────────────────────────────┤
│  CacheManager.get("/api/products/123")                           │
│    │                                                              │
│    ├─ HIT → Return cached response (10ms)                        │
│    │         Reward: +10                                         │
│    │         [END]                                               │
│    │                                                              │
│    └─ MISS → Continue to Step 2                                  │
│              Reward: -1                                           │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: Backend Request                                         │
├─────────────────────────────────────────────────────────────────┤
│  Forward request to Product Service                              │
│    • Latency: 50-200ms                                           │
│    • May trigger cascade if overloaded                           │
│    • Response cached automatically                               │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: Observe API Call                                        │
├─────────────────────────────────────────────────────────────────┤
│  MarkovPredictor.observe("/api/products/123", context)           │
│    • Update transition probabilities                             │
│    • Maintain recent history window                              │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: Get Predictions                                         │
├─────────────────────────────────────────────────────────────────┤
│  predictions = MarkovPredictor.predict(k=5)                      │
│    • Returns: [("/api/cart/add", 0.75),                          │
│                ("/api/products/456", 0.15),                      │
│                ("/api/profile", 0.08), ...]                      │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: Build State Vector                                      │
├─────────────────────────────────────────────────────────────────┤
│  state = StateBuilder.build(                                     │
│      markov_predictions,                                         │
│      cache_metrics,                                              │
│      system_metrics,                                             │
│      context                                                     │
│  )                                                               │
│  → 60-dimensional vector [0.0 ... 1.0]                           │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 6: Agent Selects Action                                    │
├─────────────────────────────────────────────────────────────────┤
│  action = DQNAgent.select_action(state)                          │
│    • Forward pass through Q-network                              │
│    • Epsilon-greedy exploration                                  │
│    • Returns action ID (0-6)                                     │
│                                                                   │
│  Possible actions:                                               │
│    0: DO_NOTHING                                                 │
│    1: CACHE_ITEM                                                 │
│    2: EVICT_LRU                                                  │
│    3: EVICT_MARKOV                                               │
│    4: PREFETCH_TOP1                                              │
│    5: PREFETCH_TOP3                                              │
│    6: PREFETCH_TOP5                                              │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 7: Execute Action                                          │
├─────────────────────────────────────────────────────────────────┤
│  Example: PREFETCH_TOP3                                          │
│    CacheManager.prefetch([                                       │
│        "/api/cart/add",                                          │
│        "/api/products/456",                                      │
│        "/api/profile"                                            │
│    ])                                                            │
│    • Fetch from backend asynchronously                           │
│    • Store in cache with TTL                                     │
│    • Track prefetch metrics                                      │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 8: Update System Metrics                                   │
├─────────────────────────────────────────────────────────────────┤
│  • Cache utilization                                             │
│  • Hit/miss counters                                             │
│  • Latency statistics                                            │
│  • Cascade risk score                                            │
│  • Bandwidth usage                                               │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 9: Calculate Reward                                        │
├─────────────────────────────────────────────────────────────────┤
│  reward = RewardCalculator.calculate(                            │
│      cache_result,                                               │
│      latency,                                                    │
│      cascade_risk,                                               │
│      prefetch_efficiency,                                        │
│      bandwidth                                                   │
│  )                                                               │
│  → Single scalar value [-100, 100]                               │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 10: Store Experience                                       │
├─────────────────────────────────────────────────────────────────┤
│  ReplayBuffer.store(                                             │
│      state=old_state,                                            │
│      action=selected_action,                                     │
│      reward=calculated_reward,                                   │
│      next_state=new_state,                                       │
│      done=episode_terminated                                     │
│  )                                                               │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 11: Train Agent (if buffer sufficient)                     │
├─────────────────────────────────────────────────────────────────┤
│  if len(replay_buffer) > batch_size:                             │
│      batch = replay_buffer.sample(64)                            │
│      loss = DQNAgent.train_step(batch)                           │
│      • Compute target Q-values                                   │
│      • Update online network weights                             │
│      • Periodically update target network                        │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RETURN RESPONSE TO CLIENT                     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Training Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  INITIALIZATION                                                   │
├─────────────────────────────────────────────────────────────────┤
│  1. Load configuration                                           │
│  2. Create Gymnasium environment                                 │
│  3. Initialize DQN agent (random weights)                        │
│  4. Initialize replay buffer (empty)                             │
│  5. Load pre-trained Markov model (if available)                 │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  FOR EACH EPISODE (1 to num_episodes)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. RESET ENVIRONMENT                                      │  │
│  │     state, info = env.reset()                              │  │
│  │     • Generate new user session                            │  │
│  │     • Reset episode metrics                                │  │
│  │     • Clear Markov history                                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                        │                                          │
│                        ▼                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  2. EPISODE LOOP (until done)                              │  │
│  │                                                            │  │
│  │  WHILE not terminated and not truncated:                   │  │
│  │                                                            │  │
│  │    ┌─────────────────────────────────────────────────┐    │  │
│  │    │ a) Select Action                                 │    │  │
│  │    │    action = agent.select_action(state)           │    │  │
│  │    └─────────────────────────────────────────────────┘    │  │
│  │              │                                              │  │
│  │              ▼                                              │  │
│  │    ┌─────────────────────────────────────────────────┐    │  │
│  │    │ b) Execute Step                                  │    │  │
│  │    │    next_state, reward, term, trunc, info =       │    │  │
│  │    │        env.step(action)                          │    │  │
│  │    └─────────────────────────────────────────────────┘    │  │
│  │              │                                              │  │
│  │              ▼                                              │  │
│  │    ┌─────────────────────────────────────────────────┐    │  │
│  │    │ c) Store Transition                              │    │  │
│  │    │    agent.store_transition(                       │    │  │
│  │    │        state, action, reward,                    │    │  │
│  │    │        next_state, term or trunc                 │    │  │
│  │    │    )                                             │    │  │
│  │    └─────────────────────────────────────────────────┘    │  │
│  │              │                                              │  │
│  │              ▼                                              │  │
│  │    ┌─────────────────────────────────────────────────┐    │  │
│  │    │ d) Train (if enough data)                        │    │  │
│  │    │    if len(buffer) > batch_size:                  │    │  │
│  │    │        loss = agent.train_step()                 │    │  │
│  │    └─────────────────────────────────────────────────┘    │  │
│  │              │                                              │  │
│  │              ▼                                              │  │
│  │    state = next_state                                      │  │
│  │    step_count += 1                                         │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                        │                                          │
│                        ▼                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  3. LOG EPISODE RESULTS                                    │  │
│  │     • Total reward                                         │  │
│  │     • Hit rate                                             │  │
│  │     • Average latency                                      │  │
│  │     • Cascades occurred                                    │  │
│  │     • Current epsilon                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                        │                                          │
│                        ▼                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  4. CHECKPOINT (if best performance)                       │  │
│  │     if episode_reward > best_reward:                       │  │
│  │         agent.save_model("best_model.pth")                 │  │
│  │         best_reward = episode_reward                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                        │                                          │
│                        ▼                                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  5. DECAY EPSILON                                          │  │
│  │     epsilon = max(epsilon_min, epsilon * decay_rate)       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  FINAL EVALUATION                                                │
├─────────────────────────────────────────────────────────────────┤
│  1. Load best model                                              │
│  2. Run evaluation episodes (deterministic policy)               │
│  3. Compare with baselines                                       │
│  4. Generate performance report                                  │
│  5. Save trained model and results                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Deployment Architecture

### 5.1 Production Deployment Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         LOAD BALANCER                            │
│                  (NGINX / HAProxy / AWS ELB)                     │
│                                                                   │
│  • Round-robin / Least connections                               │
│  • Health checks                                                 │
│  • SSL termination                                               │
└───────────────────────┬─────────────────────────────────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
┌──────────────────┐        ┌──────────────────┐
│  API Gateway +   │        │  API Gateway +   │
│  Cache Instance  │        │  Cache Instance  │
│      (Pod 1)     │        │      (Pod 2)     │
├──────────────────┤        ├──────────────────┤
│                  │        │                  │
│ ┌──────────────┐ │        │ ┌──────────────┐ │
│ │  Controller  │ │        │ │  Controller  │ │
│ └──────────────┘ │        │ └──────────────┘ │
│        │         │        │        │         │
│        ▼         │        │        ▼         │
│ ┌──────────────┐ │        │ ┌──────────────┐ │
│ │ Cache Manager│ │        │ │ Cache Manager│ │
│ └──────────────┘ │        │ └──────────────┘ │
│        │         │        │        │         │
│        ▼         │        │        ▼         │
│ ┌──────────────┐ │        │ ┌──────────────┐ │
│ │ Markov       │ │        │ │ Markov       │ │
│ │ Predictor    │ │        │ │ Predictor    │ │
│ └──────────────┘ │        │ └──────────────┘ │
│        │         │        │        │         │
│        ▼         │        │        ▼         │
│ ┌──────────────┐ │        │ ┌──────────────┐ │
│ │ DQN Agent    │ │        │ │ DQN Agent    │ │
│ │ (Inference)  │ │        │ │ (Inference)  │ │
│ └──────────────┘ │        │ └──────────────┘ │
│                  │        │                  │
└────────┬─────────┘        └────────┬─────────┘
         │                           │
         └───────────┬───────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       REDIS CLUSTER                              │
│                   (Shared Cache Storage)                         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Redis       │  │  Redis       │  │  Redis       │          │
│  │  Master      │→ │  Replica 1   │  │  Replica 2   │          │
│  │              │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  • High availability (failover)                                  │
│  • Data replication                                              │
│  • Persistence (RDB + AOF)                                       │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKEND MICROSERVICES                          │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Auth    │  │  User    │  │ Product  │  │  Order   │        │
│  │ Service  │  │ Service  │  │ Service  │  │ Service  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     MONITORING & LOGGING                         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Prometheus   │  │  Grafana     │  │ ELK Stack    │          │
│  │ (Metrics)    │  │ (Dashboard)  │  │ (Logs)       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING INFRASTRUCTURE                      │
│                      (Separate from Production)                  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Training Pipeline                                        │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │   │
│  │  │ Data       │→ │ Gymnasium  │→ │ DQN Agent  │         │   │
│  │  │ Collection │  │ Environment│  │ Training   │         │   │
│  │  └────────────┘  └────────────┘  └────────────┘         │   │
│  │         │                                │                │   │
│  │         └────────────────────────────────┘                │   │
│  │                        │                                  │   │
│  │                        ▼                                  │   │
│  │              ┌────────────────┐                           │   │
│  │              │ Model Registry │                           │   │
│  │              │ (Trained Models)│                          │   │
│  │              └────────────────┘                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Deployment Configurations

**Development Environment:**
```yaml
components:
  - Cache: InMemory
  - Simulator: Local
  - Agent: Training mode
  - Monitoring: Basic logging
resources:
  - CPU: 2 cores
  - RAM: 4GB
  - Storage: 10GB
```

**Staging Environment:**
```yaml
components:
  - Cache: Single Redis instance
  - Simulator: Realistic workload
  - Agent: Evaluation mode
  - Monitoring: Full stack
resources:
  - CPU: 4 cores
  - RAM: 8GB
  - Redis: 2GB
  - Storage: 50GB
```

**Production Environment:**
```yaml
components:
  - Cache: Redis Cluster (3 nodes)
  - Load Balancer: 1 instance
  - API Gateway+Cache: 2+ instances
  - Agent: Inference only
  - Monitoring: Prometheus + Grafana
resources:
  - Cache instance: 4 CPU, 16GB RAM
  - Redis node: 8 CPU, 32GB RAM
  - Storage: 500GB (logs + models)
scaling:
  - Horizontal: Auto-scale based on load
  - Cache: Can add more instances
  - Redis: Can add more shards
```

---

## 6. User Interface Components

### 6.1 Monitoring Dashboard

```
┌─────────────────────────────────────────────────────────────────┐
│                    CACHE PERFORMANCE DASHBOARD                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  REAL-TIME METRICS                                           ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │                                                              ││
│  │  Hit Rate:        ████████████████░░░░  85.2%               ││
│  │  Miss Rate:       ███░░░░░░░░░░░░░░░░  14.8%               ││
│  │  Eviction Rate:   █░░░░░░░░░░░░░░░░░░   3.5%               ││
│  │                                                              ││
│  │  Prefetch Efficiency: 78.3%                                 ││
│  │  Average Latency:     12ms                                  ││
│  │  Cascade Risk:        Low (0.12)                            ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  CACHE UTILIZATION (Time Series - Last 24h)                 ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  100%│                                ╭───╮                 ││
│  │      │                         ╭──────╯   ╰──╮              ││
│  │   75%│                  ╭──────╯              ╰───╮         ││
│  │      │            ╭─────╯                         ╰───╮     ││
│  │   50%│      ╭─────╯                                   ╰──   ││
│  │      │╭─────╯                                              ││
│  │   25%│                                                      ││
│  │      └────┬────┬────┬────┬────┬────┬────┬────┬────┬────   ││
│  │         00:00 04:00 08:00 12:00 16:00 20:00 24:00         ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  ACTION DISTRIBUTION (Last 1000 Requests)                    ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │                                                              ││
│  │  PREFETCH_TOP3      ████████████████████░░░░  35%           ││
│  │  CACHE_ITEM         ███████████████░░░░░░░░░  28%           ││
│  │  DO_NOTHING         ███████████░░░░░░░░░░░░░  20%           ││
│  │  PREFETCH_TOP1      █████░░░░░░░░░░░░░░░░░░░  10%           ││
│  │  EVICT_MARKOV       ██░░░░░░░░░░░░░░░░░░░░░░   5%           ││
│  │  PREFETCH_TOP5      █░░░░░░░░░░░░░░░░░░░░░░░   2%           ││
│  │  EVICT_LRU          ░░░░░░░░░░░░░░░░░░░░░░░░   0%           ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  AGENT PERFORMANCE                                           ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │                                                              ││
│  │  Episode Reward:    350.2                                   ││
│  │  Epsilon (ε):       0.05                                    ││
│  │  Q-Value (avg):     45.3                                    ││
│  │  Loss:              0.012                                   ││
│  │                                                              ││
│  │  Model Version:     v2.3.1                                  ││
│  │  Last Updated:      2026-02-04 14:30:00                     ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  ALERTS & WARNINGS                                           ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │                                                              ││
│  │  ✅ All systems operational                                  ││
│  │  ℹ️ Cache utilization high (92%) - Consider scaling          ││
│  │                                                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Configuration Interface

```
┌─────────────────────────────────────────────────────────────────┐
│                   SYSTEM CONFIGURATION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CACHE SETTINGS:                                                 │
│  ├─ Backend:           [Redis ▼]                                │
│  ├─ Capacity:          [1000    ] items                         │
│  ├─ Default TTL:       [300     ] seconds                       │
│  └─ Compression:       [✓] Enable for objects > 1KB             │
│                                                                   │
│  MARKOV SETTINGS:                                                │
│  ├─ Model Order:       [● First  ○ Second  ○ Context-Aware]     │
│  ├─ Smoothing:         [Laplace ▼]                              │
│  └─ Update Frequency:  [Real-time ▼]                            │
│                                                                   │
│  RL AGENT SETTINGS:                                              │
│  ├─ Algorithm:         [DQN ▼]                                  │
│  ├─ Learning Rate:     [0.001   ]                               │
│  ├─ Epsilon:           [0.05    ]                               │
│  ├─ Batch Size:        [64      ]                               │
│  └─ Target Update:     [1000    ] steps                         │
│                                                                   │
│  REWARD WEIGHTS:                                                 │
│  ├─ Cache Hit:         [+10.0   ]                               │
│  ├─ Cache Miss:        [-1.0    ]                               │
│  ├─ Cascade Prevented: [+50.0   ]                               │
│  └─ Cascade Occurred:  [-100.0  ]                               │
│                                                                   │
│  [Save Configuration]  [Reset to Defaults]                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Integration Points

### 7.1 External Integrations

**REST API:**
```python
# Cache operations endpoint
POST /api/cache/get
POST /api/cache/set
DELETE /api/cache/delete

# Metrics endpoint
GET /api/metrics

# Configuration endpoint
GET /api/config
PUT /api/config

# Health check
GET /health
```

**Redis Protocol:**
- Standard Redis commands (GET, SET, DELETE, TTL)
- Redis Cluster protocol for distributed deployment
- Redis pub/sub for cache invalidation

**OpenAI Gymnasium:**
- Standard Env interface
- Compatible with stable-baselines3, RLlib, etc.
- Custom observation/action spaces

**Prometheus Metrics:**
- Counter metrics (hits, misses, evictions)
- Gauge metrics (utilization, latency)
- Histogram metrics (request duration distribution)

**TensorBoard:**
- Training metrics logging
- Real-time visualization during training
- Model graph visualization

---

## 8. Security Architecture

### 8.1 Security Layers

**Application Security:**
- Input validation on all API endpoints
- Rate limiting to prevent abuse
- Error handling without information leakage

**Data Security:**
- Encryption at rest (Redis with encryption)
- Encryption in transit (TLS/SSL)
- Sensitive data anonymization in logs

**Access Control:**
- API authentication (when integrated with auth service)
- Role-based access control for configuration
- Audit logging of all admin operations

**Network Security:**
- Private subnets for backend communication
- Firewall rules restricting access
- DDoS protection at load balancer

---

## 9. Scalability Architecture

### 9.1 Horizontal Scaling

**Cache Layer:**
- Multiple cache instances share Redis cluster
- Load balancer distributes requests
- Stateless design enables easy scaling

**Redis Layer:**
- Redis Cluster with sharding
- Can add more shards for capacity
- Automatic failover and replication

**Microservices:**
- Independent scaling of backend services
- Container orchestration (Kubernetes)
- Auto-scaling based on metrics

### 9.2 Performance Optimizations

**Caching:**
- Multi-level caching (L1: local, L2: Redis)
- Cache warming on startup
- Batch prefetching to reduce roundtrips

**Computation:**
- Vectorized operations (NumPy)
- Batched RL inference
- GPU acceleration for training (optional)

**Network:**
- Connection pooling
- Persistent connections to Redis
- Compression for large payloads

---

**End of Architecture Documentation**
