# Chapter 6: Design

## 6.1 Chapter Overview

### Purpose

The design phase represents a critical stage in the development of the Markov Chain-based Reinforcement Learning framework for adaptive API caching in microservices. This chapter articulates the systematic transformation of system requirements, identified in the Software Requirements Specification (SRS), into a comprehensive architectural and detailed design that guides the implementation process. The design decisions documented herein establish the foundation for creating a scalable, maintainable, and high-performance intelligent caching system.

### Importance to the Project

The design phase serves multiple essential functions in this project:

1. **Blueprint for Implementation**: Provides a clear roadmap that translates abstract requirements into concrete structural and behavioural specifications.

2. **Risk Mitigation**: Identifies potential architectural bottlenecks, integration challenges, and performance constraints before implementation commences.

3. **Stakeholder Communication**: Offers visual and textual representations that facilitate understanding among technical and non-technical stakeholders.

4. **Quality Assurance Foundation**: Establishes measurable design goals and constraints that enable systematic validation of the implementation.

5. **Maintenance and Evolution**: Creates a documented reference architecture that supports future enhancements and modifications.

### Chapter Structure

This chapter is organised into six main sections:

- **Section 6.2: Design Goals** defines the overarching quality attributes that the system must satisfy, derived directly from the non-functional requirements (NFRs) specified in the SRS.

- **Section 6.3: System Architecture Diagram** presents the high-level tiered architecture, explaining how major components interact within the distributed system.

- **Section 6.4: Detailed Design** provides component-level design diagrams, including class diagrams, sequence diagrams, and data flow diagrams that specify the internal structure and behaviour of key subsystems.

- **Section 6.5: Algorithm Design** articulates the reinforcement learning algorithms and Markov chain prediction mechanisms through pseudocode and flowcharts.

- **Section 6.6: Neural Network Architecture** details the Deep Q-Network (DQN) structure used for learning optimal caching policies.

- **Section 6.7: Chapter Summary** synthesises the design decisions and their rationale.

The design presented in this chapter adheres to Object-Oriented Analysis and Design Methodology (OOADM), employing industry-standard UML notation where appropriate. This approach was selected due to the object-oriented nature of the implementation languages (Python) and the need for clear encapsulation of complex machine learning and distributed systems components.

---

## 6.2 Design Goals

The design goals represent key quality attributes that define the characteristics of the software system. These goals are derived from the Non-Functional Requirements (NFRs) specified in the SRS and are maintained consistently throughout the system architecture and implementation.

### DG1: Performance

**Definition**: The system must process API requests and make caching decisions with minimal latency overhead whilst maximising cache hit rates.

**Rationale**: In microservices architectures, every millisecond of latency compounds across service dependencies. The caching system must enhance, not degrade, overall system performance.

**Design Implications**:
- Asynchronous processing for non-blocking cache operations
- In-memory data structures for rapid state representation
- Optimised Redis connections with connection pooling
- Batch processing capabilities for prediction operations
- Pre-computed Markov transition matrices

**Measurement Criteria**:
- Cache decision latency < 5ms (p95)
- Cache hit rate improvement > 15% compared to LRU baseline
- API response time reduction of 30-50% for cached responses

### DG2: Scalability

**Definition**: The system must accommodate increasing API traffic volumes, expanding cache sizes, and growing numbers of microservices without architectural redesign.

**Rationale**: Enterprise microservices ecosystems are dynamic and continuously evolving. The caching framework must scale horizontally and vertically.

**Design Implications**:
- Stateless service design enabling horizontal scaling
- Distributed cache backend (Redis cluster support)
- Modular architecture allowing independent component scaling
- Efficient memory management with configurable cache size limits
- Partitionable observation spaces for distributed learning

**Measurement Criteria**:
- Linear scalability up to 10,000 requests/second
- Support for cache sizes up to 10GB
- Graceful degradation under load

### DG3: Accuracy

**Definition**: The reinforcement learning agent and Markov predictors must make accurate predictions about future API calls to enable effective prefetching and eviction decisions.

**Rationale**: Prediction accuracy directly impacts cache hit rates and system efficiency. Inaccurate predictions waste resources through unnecessary prefetching and premature evictions.

**Design Implications**:
- Hybrid prediction approach combining Markov chains and DQN
- Context-aware prediction incorporating temporal and user-type features
- Continuous model updating based on recent observations
- Ensemble prediction with confidence estimation
- Separate first-order and second-order Markov models

**Measurement Criteria**:
- Prediction accuracy > 75% for next API call (Markov models)
- DQN convergence to near-optimal policy within 1000 episodes
- Prefetch precision > 60% (prefetched items subsequently requested)

### DG4: Maintainability

**Definition**: The system codebase must be modular, well-documented, and structured to facilitate understanding, debugging, and enhancement by developers.

**Rationale**: Academic and production software requires long-term maintenance. Clear structure and documentation reduce technical debt and support knowledge transfer.

**Design Implications**:
- Modular architecture with clear separation of concerns
- Comprehensive API documentation for all public interfaces
- Consistent coding standards and naming conventions
- Extensive inline documentation and type hints
- Standardised logging and error handling mechanisms

**Measurement Criteria**:
- Code coverage > 80% with unit and integration tests
- Complete API documentation for all modules
- Adherence to PEP 8 style guidelines (Python)
- Average cyclomatic complexity < 10 per function

### DG5: Reliability and Fault Tolerance

**Definition**: The system must continue operating correctly even when components fail or unexpected conditions arise, with graceful degradation rather than catastrophic failure.

**Rationale**: Caching is a performance optimisation, not a correctness requirement. Cache failures must not cause application failures.

**Design Implications**:
- Exception handling at all integration points
- Fallback mechanisms to simpler caching strategies
- Cascade failure detection with circuit breakers
- Comprehensive logging for debugging and auditing
- Timeout mechanisms preventing indefinite blocking

**Measurement Criteria**:
- System uptime > 99.9%
- Graceful handling of Redis connection failures
- No data loss during component restarts
- Recovery time < 30 seconds after failure

### DG6: Extensibility

**Definition**: The architecture must accommodate future enhancements such as additional RL algorithms, alternative cache backends, or new prediction models without requiring significant refactoring.

**Rationale**: Research and production systems evolve. The design must anticipate change and minimize coupling.

**Design Implications**:
- Abstract interfaces for cache backends, predictors, and RL agents
- Plugin architecture for reward functions and state representations
- Configuration-driven behaviour reducing hardcoded dependencies
- Strategy pattern for algorithm selection
- Factory pattern for component instantiation

**Measurement Criteria**:
- New cache backend integration < 2 days of development
- New RL algorithm integration < 3 days of development
- Addition of new state features without modifying core components

---

## 6.3 System Architecture Diagram

### Overview

The system employs a **four-tier layered architecture** that promotes separation of concerns, modularity, and scalability. This architectural pattern is well-suited for distributed machine learning systems where different layers have distinct responsibilities and may scale independently.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION / API TIER                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │           FastAPI Gateway / REST API Endpoints                    │  │
│  │  • /api/predict                  • /api/cache/get                │  │
│  │  • /api/cache/set                • /api/metrics                  │  │
│  │  • /api/train                    • /api/health                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATION / CONTROL TIER                       │
│  ┌────────────────────┐  ┌──────────────────────┐  ┌────────────────┐ │
│  │   Controller       │  │  Gym Environment     │  │  Metrics       │ │
│  │   • Request Router │  │  • Episode Manager   │  │  • Monitoring  │ │
│  │   • Action Decode  │  │  • State Builder     │  │  • Evaluation  │ │
│  │   • Orchestration  │  │  • Reward Calculator │  │  • Analysis    │ │
│  └────────────────────┘  └──────────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│     BUSINESS LOGIC TIER         │  │     MACHINE LEARNING TIER       │
│  ┌──────────────────────────┐   │  │  ┌──────────────────────────┐  │
│  │   Cache Manager          │   │  │  │   DQN Agent              │  │
│  │   • Cache Operations     │   │  │  │   • Q-Network (PyTorch)  │  │
│  │   • Eviction Policies    │   │  │  │   • Action Selection     │  │
│  │   • Prefetch Logic       │   │  │  │   • Training Loop        │  │
│  └──────────────────────────┘   │  │  └──────────────────────────┘  │
│  ┌──────────────────────────┐   │  │  ┌──────────────────────────┐  │
│  │   Markov Predictors      │   │  │  │   Replay Buffer          │  │
│  │   • First-Order          │   │  │  │   • Experience Storage   │  │
│  │   • Second-Order         │   │  │  │   • Sampling Strategy    │  │
│  │   • Context-Aware        │   │  │  └──────────────────────────┘  │
│  │   • Transition Matrix    │   │  │  ┌──────────────────────────┐  │
│  └──────────────────────────┘   │  │  │   Trainer                │  │
│                                  │  │  │   • Optimization         │  │
│                                  │  │  │   • Loss Calculation     │  │
│                                  │  │  └──────────────────────────┘  │
└─────────────────────────────────┘  └─────────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA / PERSISTENCE TIER                          │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │  Redis Cache   │  │  Model Storage  │  │  Training Data          │ │
│  │  • Key-Value   │  │  • Q-Network    │  │  • API Logs (Parquet)   │ │
│  │  • TTL         │  │    Checkpoints  │  │  • Session Sequences    │ │
│  │  • Clustering  │  │  • Best Models  │  │  • Transition Matrices  │ │
│  └────────────────┘  └─────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer Descriptions

#### Tier 1: Presentation / API Layer

**Purpose**: Provides the external interface for client applications and system administrators to interact with the caching framework.

**Components**:
- **FastAPI Gateway**: RESTful API server handling HTTP requests and responses
- **Request Validation**: Input validation and sanitisation using Pydantic models
- **Authentication/Authorization**: (Future) Security layer for production deployment

**Responsibilities**:
- Expose cache operations (get, set, delete) via REST endpoints
- Provide prediction API for proactive cache warming
- Serve metrics and monitoring data for observability
- Handle training triggers and model management requests

**Technologies**: FastAPI, Uvicorn (ASGI server), Pydantic

#### Tier 2: Integration / Control Layer

**Purpose**: Orchestrates interactions between the presentation layer, business logic, and machine learning components. This tier implements the control flow and decision-making logic.

**Components**:
- **Controller**: Central orchestration component managing request routing and action execution
- **Gym Environment**: OpenAI Gymnasium-compliant environment for RL training
- **Metrics Aggregator**: Collects and processes performance metrics

**Responsibilities**:
- Decode high-level API requests into specific cache operations
- Manage episode lifecycle during training (reset, step, termination)
- Construct state observations from system state
- Calculate rewards based on cache performance
- Coordinate between ML predictions and cache operations
- Track and report system metrics

**Technologies**: Python (asyncio for asynchronous operations), Gymnasium API

#### Tier 3a: Business Logic Layer

**Purpose**: Implements core caching functionality and domain-specific logic independent of machine learning components.

**Components**:
- **Cache Manager**: High-level cache interface abstracting backend details
- **Prefetch Engine**: Implements intelligent prefetching based on predictions
- **Markov Predictors**: Family of prediction models for API call sequences

**Responsibilities**:
- Execute cache CRUD operations (Create, Read, Update, Delete)
- Implement eviction policies (LRU, probability-based, hybrid)
- Perform prefetching based on Markov predictions
- Maintain transition matrices for first and second-order Markov chains
- Generate context-aware predictions incorporating user types and time features

**Technologies**: Python, NumPy (for matrix operations), SciPy

#### Tier 3b: Machine Learning Layer

**Purpose**: Houses reinforcement learning components responsible for learning optimal caching policies through interaction with the environment.

**Components**:
- **DQN Agent**: Implements Deep Q-Learning algorithm
- **Q-Network**: Neural network approximating action-value function Q(s,a)
- **Replay Buffer**: Stores and samples experience tuples for training
- **Trainer**: Manages training loop, optimization, and model updates

**Responsibilities**:
- Select actions using ε-greedy policy (exploration vs exploitation)
- Train Q-Network using experience replay and target network
- Store transitions (state, action, reward, next_state) in replay buffer
- Optimize network parameters using gradient descent
- Manage training hyperparameters and learning schedules

**Technologies**: PyTorch (neural networks), NumPy, Python

#### Tier 4: Data / Persistence Layer

**Purpose**: Provides persistent storage for cached data, trained models, and training datasets.

**Components**:
- **Redis Cache Backend**: Distributed in-memory cache
- **Model Storage**: File system persistence for neural network checkpoints
- **Training Data Repository**: Parquet files containing API logs and sequences

**Responsibilities**:
- Store and retrieve cached API responses with TTL management
- Persist trained Q-Network weights for inference and continued training
- Maintain historical API request logs for offline training
- Store preprocessed session sequences and feature-engineered data
- Save transition matrices for Markov predictors

**Technologies**: Redis (in-memory cache), PyTorch model serialisation, Parquet (columnar storage)

### Architectural Patterns

The architecture employs several design patterns:

1. **Layered Architecture**: Clear separation between presentation, business logic, ML, and data layers
2. **Strategy Pattern**: Interchangeable caching strategies (LRU, LFU, ML-based)
3. **Observer Pattern**: Metrics collection and monitoring
4. **Factory Pattern**: Component instantiation based on configuration
5. **Adapter Pattern**: Uniform interface for different cache backends (Redis, in-memory)

### Data Flow

```
[Client Request] 
    → [FastAPI Gateway] 
    → [Controller] 
    → [Markov Predictor.predict()] 
    → [DQN Agent.select_action()] 
    → [Cache Manager.execute(action)] 
    → [Redis Backend] 
    → [Response] 
    → [Reward Calculation] 
    → [State Update] 
    → [Replay Buffer.store()] 
    → [Trainer.train()]
```

---

## 6.4 Detailed Design

This section presents detailed design diagrams for key subsystems, illustrating the internal structure and interactions of components. The diagrams follow Object-Oriented Analysis and Design Methodology (OOADM) using UML notation.

### 6.4.1 Class Diagram: Cache Management Subsystem

The cache management subsystem is responsible for interfacing with the Redis backend and implementing various caching strategies.

```
┌─────────────────────────────────────────────────────────────┐
│                    <<interface>>                             │
│                    CacheBackend                              │
├─────────────────────────────────────────────────────────────┤
│ + get(key: str) → Optional[Any]                            │
│ + set(key: str, value: Any, ttl: int) → bool               │
│ + delete(key: str) → bool                                   │
│ + exists(key: str) → bool                                   │
│ + get_size() → int                                          │
│ + clear() → None                                            │
└─────────────────────────────────────────────────────────────┘
                         ▲
                         │ implements
         ┌───────────────┴───────────────┐
         │                               │
┌────────────────────┐      ┌────────────────────────┐
│  RedisBackend      │      │  InMemoryBackend       │
├────────────────────┤      ├────────────────────────┤
│ - client: Redis    │      │ - cache: Dict          │
│ - host: str        │      │ - ttl_map: Dict        │
│ - port: int        │      │ - max_size: int        │
├────────────────────┤      ├────────────────────────┤
│ + __init__(...)    │      │ + __init__(...)        │
│ + get(key)         │      │ + get(key)             │
│ + set(key, val)    │      │ + set(key, val)        │
│ + delete(key)      │      │ + delete(key)          │
│ + _check_ttl()     │      │ + _evict_expired()     │
└────────────────────┘      └────────────────────────┘
                         
                         
┌─────────────────────────────────────────────────────────────┐
│                    CacheManager                              │
├─────────────────────────────────────────────────────────────┤
│ - backend: CacheBackend                                     │
│ - prefetch_engine: PrefetchEngine                           │
│ - metrics: MetricsCollector                                 │
│ - config: CacheConfig                                       │
├─────────────────────────────────────────────────────────────┤
│ + __init__(backend, config)                                 │
│ + get(key: str) → Optional[Any]                            │
│ + set(key: str, value: Any) → None                         │
│ + prefetch(keys: List[str]) → None                         │
│ + evict_lru() → None                                        │
│ + evict_by_probability(threshold: float) → None            │
│ + get_cache_state() → Dict                                  │
│ + record_access(key: str) → None                           │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 PrefetchEngine                               │
├─────────────────────────────────────────────────────────────┤
│ - predictor: MarkovPredictor                                │
│ - backend: CacheBackend                                     │
│ - max_prefetch: int                                         │
├─────────────────────────────────────────────────────────────┤
│ + prefetch_based_on_prediction(current_api: str) → int     │
│ + validate_prefetch_accuracy() → float                      │
└─────────────────────────────────────────────────────────────┘
```

**Key Relationships**:
- `CacheManager` depends on `CacheBackend` interface (Dependency Injection)
- `RedisBackend` and `InMemoryBackend` implement `CacheBackend` (Interface Implementation)
- `CacheManager` has a `PrefetchEngine` (Composition)
- `PrefetchEngine` uses `MarkovPredictor` (Association)

### 6.4.2 Class Diagram: Reinforcement Learning Subsystem

The RL subsystem implements the DQN algorithm for learning optimal caching policies.

```
┌─────────────────────────────────────────────────────────────┐
│                       QNetwork                               │
│                   (torch.nn.Module)                          │
├─────────────────────────────────────────────────────────────┤
│ - fc1: nn.Linear                                            │
│ - fc2: nn.Linear                                            │
│ - fc3: nn.Linear                                            │
│ - dropout: nn.Dropout                                       │
│ - state_dim: int                                            │
│ - action_dim: int                                           │
├─────────────────────────────────────────────────────────────┤
│ + __init__(state_dim, action_dim, hidden_dim)              │
│ + forward(state: Tensor) → Tensor                          │
└─────────────────────────────────────────────────────────────┘
                         ▲
                         │ uses
                         │
┌─────────────────────────────────────────────────────────────┐
│                      DQNAgent                                │
├─────────────────────────────────────────────────────────────┤
│ - q_network: QNetwork                                       │
│ - target_network: QNetwork                                  │
│ - replay_buffer: ReplayBuffer                               │
│ - epsilon: float                                            │
│ - gamma: float                                              │
│ - learning_rate: float                                      │
│ - device: torch.device                                      │
├─────────────────────────────────────────────────────────────┤
│ + __init__(state_dim, action_dim, config)                  │
│ + select_action(state: np.ndarray) → int                   │
│ + store_transition(s, a, r, s', done) → None              │
│ + train_step() → float                                      │
│ + update_target_network() → None                           │
│ + save_model(path: str) → None                            │
│ + load_model(path: str) → None                            │
│ + get_epsilon() → float                                     │
│ + decay_epsilon(decay_rate: float) → None                  │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ uses
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    ReplayBuffer                              │
├─────────────────────────────────────────────────────────────┤
│ - buffer: deque                                             │
│ - max_size: int                                             │
│ - batch_size: int                                           │
├─────────────────────────────────────────────────────────────┤
│ + __init__(max_size, batch_size)                           │
│ + store(state, action, reward, next_state, done) → None   │
│ + sample() → Tuple[Tensors]                                │
│ + __len__() → int                                           │
│ + is_ready() → bool                                         │
└─────────────────────────────────────────────────────────────┘
                         

┌─────────────────────────────────────────────────────────────┐
│                       Trainer                                │
├─────────────────────────────────────────────────────────────┤
│ - agent: DQNAgent                                           │
│ - env: gym.Env                                              │
│ - num_episodes: int                                         │
│ - log_interval: int                                         │
│ - save_interval: int                                        │
├─────────────────────────────────────────────────────────────┤
│ + train() → Dict[str, Any]                                  │
│ + evaluate(num_episodes: int) → Dict[str, float]           │
│ + save_checkpoint(episode: int) → None                     │
│ + load_checkpoint(path: str) → None                        │
└─────────────────────────────────────────────────────────────┘
```

### 6.4.3 Sequence Diagram: Cache Request Processing

This diagram illustrates the sequence of interactions when processing an API cache request with ML-based decision making.

```
Client    Gateway    Controller   Predictor   DQN Agent   CacheManager   Redis
  │          │           │            │            │            │           │
  │─Request──>│           │            │            │            │           │
  │          │──route────>│            │            │            │           │
  │          │           │─predict────>│            │            │           │
  │          │           │<─probs─────┘            │            │           │
  │          │           │─get_state───────────────>│            │           │
  │          │           │<─state──────────────────┘            │           │
  │          │           │─select_action()──────────>│           │           │
  │          │           │<─action─────────────────┘            │           │
  │          │           │                      (action: CACHE) │           │
  │          │           │─execute(action, key)─────────────────>│           │
  │          │           │                                       │─set()────>│
  │          │           │                                       │<─ok──────┘
  │          │           │<─success────────────────────────────┘            │
  │          │           │                                                   │
  │          │           │─calculate_reward()                                │
  │          │           │                                                   │
  │          │           │─store_transition(s,a,r,s')────────────>│          │
  │          │           │                                                   │
  │          │<─response─┤                                                   │
  │<─Response─┘          │                                                   │
  │                      │                                                   │
  │          (Background Training)                                           │
  │          │           │─train_step()──────────────────────────>│          │
  │          │           │  ├─sample_batch()                                │
  │          │           │  ├─compute_loss()                                │
  │          │           │  └─backward()                                    │
  │          │           │<─loss──────────────────────────────────┘         │
```

### 6.4.4 Activity Diagram: Training Episode Flow

This diagram shows the control flow during a training episode in the Gymnasium environment.

```
                    ┌───────────┐
                    │   START   │
                    └─────┬─────┘
                          │
                          ▼
                    ┌───────────┐
                    │env.reset()│
                    └─────┬─────┘
                          │
                ┌─────────▼─────────┐
                │ Generate Session  │
                │ • User Type       │
                │ • API Sequence    │
                └─────────┬─────────┘
                          │
                          ▼
                ┌─────────────────┐
                │ Build Initial   │
                │ Observation     │
                └─────┬───────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Episode Loop  │◄────────┐
              └───────┬───────┘         │
                      │                 │
                      ▼                 │
              ┌─────────────────┐       │
              │ Get Predictions │       │
              │ (Markov)        │       │
              └────────┬────────┘       │
                       │                │
                       ▼                │
              ┌─────────────────┐       │
              │ Agent Selects   │       │
              │ Action (ε-greedy)│      │
              └────────┬────────┘       │
                       │                │
                       ▼                │
              ┌─────────────────┐       │
              │ Execute Action  │       │
              │ on Cache        │       │
              └────────┬────────┘       │
                       │                │
                       ▼                │
              ┌─────────────────┐       │
              │ Get Next API    │       │
              │ from Sequence   │       │
              └────────┬────────┘       │
                       │                │
                       ▼                │
              ┌─────────────────┐       │
         ┌────┤ Check Cache     │       │
         │    └─────────────────┘       │
         │              │               │
    ┌────▼────┐    ┌────▼────┐        │
    │ HIT     │    │ MISS    │        │
    │reward+10│    │reward-1 │        │
    └────┬────┘    └────┬────┘        │
         │              │              │
         └──────┬───────┘              │
                │                      │
                ▼                      │
        ┌───────────────┐              │
        │Calculate Total│              │
        │Reward         │              │
        └───────┬───────┘              │
                │                      │
                ▼                      │
        ┌───────────────┐              │
        │Store in Replay│              │
        │Buffer         │              │
        └───────┬───────┘              │
                │                      │
                ▼                      │
        ┌───────────────┐              │
    ┌───┤Termination?   │              │
    │   │• Cascade      │              │
    │   │• Step Limit   │              │
    │   │• Session End  │              │
    │   └───────────────┘              │
    │          │                       │
    │         NO                       │
    │          └───────────────────────┘
    │
   YES
    │
    ▼
┌─────────────────┐
│ Train Agent     │
│ (batch update)  │
└────────┬────────┘
         │
         ▼
    ┌─────────┐
    │  DONE   │
    └─────────┘
```

### 6.4.5 Component Diagram: Markov Prediction System

```
┌────────────────────────────────────────────────────────────────┐
│                 Markov Prediction System                        │
│                                                                 │
│  ┌──────────────────┐        ┌──────────────────────┐         │
│  │ FirstOrderMarkov │        │ SecondOrderMarkov    │         │
│  ├──────────────────┤        ├──────────────────────┤         │
│  │ - trans_matrix   │        │ - trans_matrix_2d    │         │
│  │ - api_vocab      │        │ - bigram_vocab       │         │
│  ├──────────────────┤        ├──────────────────────┤         │
│  │ + fit(sequences) │        │ + fit(sequences)     │         │
│  │ + predict(api)   │        │ + predict(api1,api2) │         │
│  └────────┬─────────┘        └────────┬─────────────┘         │
│           │                           │                        │
│           └───────────┬───────────────┘                        │
│                       │                                        │
│                       ▼                                        │
│            ┌──────────────────────┐                           │
│            │ ContextAwareMarkov   │                           │
│            ├──────────────────────┤                           │
│            │ - base_predictor     │                           │
│            │ - user_type_weights  │                           │
│            │ - time_weights       │                           │
│            ├──────────────────────┤                           │
│            │ + fit(sequences)     │                           │
│            │ + predict(api, ctx)  │                           │
│            │ + adjust_for_context │                           │
│            └──────────────────────┘                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 6.5 Algorithm Design

This section presents the core algorithms used in the system, expressed through pseudocode and flowcharts.

### 6.5.1 Deep Q-Network (DQN) Training Algorithm

The DQN algorithm learns the optimal action-value function Q*(s,a) through iterative experience-based learning.

```
Algorithm 1: DQN Training

Input: 
  - Environment env (Gymnasium CachingEnv)
  - Number of episodes num_episodes
  - Learning rate α
  - Discount factor γ
  - Exploration rate ε
  - Replay buffer size N
  - Batch size B
  - Target network update frequency C

Output:
  - Trained Q-network parameters θ

1:  Initialize replay buffer D with capacity N
2:  Initialize Q-network with random weights θ
3:  Initialize target Q-network with weights θ⁻ = θ
4:  steps ← 0
5:  
6:  for episode = 1 to num_episodes do
7:      state ← env.reset()
8:      episode_reward ← 0
9:      done ← False
10:     
11:     while not done do
12:         // Epsilon-greedy action selection
13:         if random() < ε then
14:             action ← random_action()
15:         else
16:             Q_values ← Q(state; θ)
17:             action ← argmax(Q_values)
18:         end if
19:         
20:         // Execute action in environment
21:         next_state, reward, done, info ← env.step(action)
22:         
23:         // Store transition in replay buffer
24:         D.store((state, action, reward, next_state, done))
25:         
26:         // Sample random minibatch from D
27:         if len(D) >= B then
28:             batch ← D.sample(B)
29:             
30:             // Compute target y for each transition in batch
31:             for each (s, a, r, s', d) in batch do
32:                 if d then  // Terminal state
33:                     y ← r
34:                 else
35:                     // Double DQN: use online network for action selection
36:                     best_action ← argmax(Q(s'; θ))
37:                     // Use target network for value estimation
38:                     y ← r + γ * Q(s', best_action; θ⁻)
39:                 end if
40:             end for
41:             
42:             // Compute loss
43:             predictions ← [Q(s, a; θ) for (s, a, _, _, _) in batch]
44:             loss ← MSE(predictions, y)
45:             
46:             // Gradient descent update
47:             θ ← θ - α * ∇θ loss
48:         end if
49:         
50:         state ← next_state
51:         episode_reward ← episode_reward + reward
52:         steps ← steps + 1
53:         
54:         // Update target network
55:         if steps mod C == 0 then
56:             θ⁻ ← θ
57:         end if
58:     end while
59:     
60:     // Decay exploration rate
61:     ε ← max(ε_min, ε * ε_decay)
62:     
63:     // Log episode results
64:     log("Episode", episode, "Reward:", episode_reward)
65: end for
66: 
67: return θ
```

### 6.5.2 Markov Chain Prediction Algorithm

```
Algorithm 2: First-Order Markov Prediction

Input:
  - Training sequences S = {s₁, s₂, ..., sₙ}
  - Current API call c
  - Top-k predictions k

Output:
  - List of (api, probability) tuples

1:  // Build transition matrix from training data
2:  Initialize transition_matrix[API_VOCAB_SIZE][API_VOCAB_SIZE] with zeros
3:  Initialize count_matrix[API_VOCAB_SIZE] with zeros
4:  
5:  for each sequence seq in S do
6:      for i = 0 to len(seq) - 2 do
7:          current_api ← seq[i]
8:          next_api ← seq[i + 1]
9:          transition_matrix[current_api][next_api] += 1
10:         count_matrix[current_api] += 1
11:     end for
12: end for
13: 
14: // Normalize to get probabilities
15: for i = 0 to API_VOCAB_SIZE do
16:     if count_matrix[i] > 0 then
17:         for j = 0 to API_VOCAB_SIZE do
18:             transition_matrix[i][j] /= count_matrix[i]
19:         end for
20:     end if
21: end for
22: 
23: // Prediction phase
24: function PREDICT(current_api, k):
25:     if current_api not in vocabulary then
26:         return uniform_distribution()
27:     end if
28:     
29:     probabilities ← transition_matrix[current_api]
30:     
31:     // Get top-k predictions
32:     top_k_indices ← argsort(probabilities, descending=True)[:k]
33:     predictions ← []
34:     
35:     for idx in top_k_indices do
36:         api ← vocabulary[idx]
37:         prob ← probabilities[idx]
38:         predictions.append((api, prob))
39:     end for
40:     
41:     return predictions
42: end function
```

### 6.5.3 Action Selection and Execution Algorithm

```
Algorithm 3: Action Selection and Execution

Input:
  - Current state s (observation)
  - DQN agent
  - Markov predictions P
  - Cache manager cache_mgr

Output:
  - Executed action
  - Reward

1:  // Get action from agent
2:  action ← agent.select_action(s)
3:  
4:  // Decode action
5:  (action_type, param) ← decode_action(action)
6:  
7:  // Execute based on action type
8:  switch action_type do
9:      case DO_NOTHING:
10:         // No cache operation
11:         pass
12:         
13:     case CACHE_CURRENT:
14:         cache_mgr.set(current_api, response)
15:         
16:     case PREFETCH_TOP1:
17:         if len(P) > 0 then
18:             api_to_prefetch ← P[0].api
19:             cache_mgr.prefetch(api_to_prefetch)
20:         end if
21:         
22:     case PREFETCH_TOP3:
23:         for i = 0 to min(3, len(P)) do
24:             cache_mgr.prefetch(P[i].api)
25:         end for
26:         
27:     case EVICT_LRU:
28:         cache_mgr.evict_lru()
29:         
30:     case EVICT_LOW_PROB:
31:         // Evict items with prediction probability < threshold
32:         threshold ← 0.1
33:         cache_mgr.evict_by_probability(P, threshold)
34:         
35: end switch
36: 
37: // Check cache for current API
38: response ← cache_mgr.get(current_api)
39: 
40: // Calculate reward
41: if response is HIT then
42:     reward ← +10.0
43: else
44:     reward ← -1.0
45:     // Fetch from backend (simulated)
46:     response ← backend_service.fetch(current_api)
47:     cache_mgr.set(current_api, response)
48: end if
49: 
50: // Add penalties for resource usage
51: cache_occupancy ← cache_mgr.get_occupancy()
52: if cache_occupancy > 0.9 then
53:     reward ← reward - 5.0
54: end if
55: 
56: return action, reward
```

### 6.5.4 Flowchart: Cache Request Decision Process

```
                    ┌──────────────┐
                    │ API Request  │
                    │  Arrives     │
                    └──────┬───────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ Build State    │
                  │ Observation    │
                  └────────┬───────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ Get Markov     │
                  │ Predictions    │
                  └────────┬───────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ DQN Agent      │
                  │ Select Action  │
                  └────────┬───────┘
                           │
              ┌────────────┴────────────┐
              │                         │
          ε < random?               ε >= random?
              │                         │
              ▼                         ▼
    ┌──────────────────┐      ┌──────────────────┐
    │ EXPLOIT:         │      │ EXPLORE:         │
    │ Best Q-value     │      │ Random Action    │
    └────────┬─────────┘      └────────┬─────────┘
             │                         │
             └────────────┬────────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │ Decode Action  │
                 └────────┬───────┘
                          │
        ┌─────────────────┼─────────────────┬───────────────┐
        │                 │                 │               │
        ▼                 ▼                 ▼               ▼
  ┌──────────┐    ┌──────────────┐  ┌──────────┐   ┌──────────┐
  │ DO_      │    │ CACHE/       │  │ EVICT    │   │ PREFETCH │
  │ NOTHING  │    │ UPDATE       │  │          │   │          │
  └─────┬────┘    └──────┬───────┘  └─────┬────┘   └────┬─────┘
        │                │                │             │
        └────────────────┴────────────────┴─────────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │ Check Cache    │
                 │ for Current API│
                 └────────┬───────┘
                          │
                ┌─────────┴─────────┐
                │                   │
           ┌────▼────┐         ┌────▼────┐
           │ CACHE   │         │ CACHE   │
           │ HIT     │         │ MISS    │
           └────┬────┘         └────┬────┘
                │                   │
                ▼                   ▼
        ┌──────────────┐    ┌──────────────┐
        │ Reward: +10  │    │ Reward: -1   │
        │ Fast Response│    │ Fetch Backend│
        └──────┬───────┘    └──────┬───────┘
               │                   │
               └─────────┬─────────┘
                         │
                         ▼
                ┌────────────────┐
                │ Apply Penalties│
                │ (if applicable)│
                └────────┬───────┘
                         │
                         ▼
                ┌────────────────┐
                │ Store Transition│
                │ in Replay Buffer│
                └────────┬───────┘
                         │
                         ▼
                ┌────────────────┐
                │ Return Response│
                │ to Client      │
                └────────────────┘
```

---

## 6.6 Neural Network Architecture

The Q-Network is the core neural network component that approximates the action-value function Q(s,a), estimating the expected cumulative reward for taking action a in state s.

### 6.6.1 Q-Network Architecture

The Q-Network is a feedforward neural network (multilayer perceptron) with the following structure:

```
INPUT LAYER (60 dimensions)
┌─────────────────────────────────────────────────────────────┐
│ State Features:                                              │
│ • Markov Predictions (10): Top-10 API probabilities         │
│ • Cache State (20): Cache occupancy, hit rate, etc.         │
│ • Request Context (10): User type, time features            │
│ • History Features (20): Recent API patterns                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
HIDDEN LAYER 1 (256 neurons)
┌─────────────────────────────────────────────────────────────┐
│ • Fully Connected (Linear): 60 → 256                        │
│ • Activation: ReLU                                           │
│ • BatchNorm1d (optional, for stability)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
DROPOUT (p=0.2)
┌─────────────────────────────────────────────────────────────┐
│ • Regularization to prevent overfitting                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
HIDDEN LAYER 2 (256 neurons)
┌─────────────────────────────────────────────────────────────┐
│ • Fully Connected (Linear): 256 → 256                       │
│ • Activation: ReLU                                           │
│ • BatchNorm1d (optional)                                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
DROPOUT (p=0.2)
                         │
                         ▼
HIDDEN LAYER 3 (128 neurons)
┌─────────────────────────────────────────────────────────────┐
│ • Fully Connected (Linear): 256 → 128                       │
│ • Activation: ReLU                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
OUTPUT LAYER (7 actions)
┌─────────────────────────────────────────────────────────────┐
│ • Fully Connected (Linear): 128 → 7                         │
│ • No activation (raw Q-values)                              │
│                                                              │
│ Actions:                                                     │
│   0: DO_NOTHING                                             │
│   1: CACHE_CURRENT                                          │
│   2: PREFETCH_TOP1                                          │
│   3: PREFETCH_TOP3                                          │
│   4: EVICT_LRU                                              │
│   5: EVICT_LOW_PROBABILITY                                  │
│   6: HYBRID_STRATEGY                                        │
└─────────────────────────────────────────────────────────────┘
```

### 6.6.2 Network Specifications

**Architecture Summary**:
- **Type**: Feedforward Neural Network (Multilayer Perceptron)
- **Input Dimension**: 60 features
- **Hidden Layers**: 3 layers (256 → 256 → 128 neurons)
- **Output Dimension**: 7 actions
- **Total Parameters**: ~100,000 trainable parameters

**Activation Functions**:
- **Hidden Layers**: ReLU (Rectified Linear Unit)
  - Formula: f(x) = max(0, x)
  - Benefits: Prevents vanishing gradients, computationally efficient
- **Output Layer**: None (linear)
  - Q-values can be positive or negative, so no activation is applied

**Regularization**:
- **Dropout**: p = 0.2 (20% of neurons randomly dropped during training)
- **Purpose**: Prevent overfitting, improve generalization

**Optimization**:
- **Loss Function**: Mean Squared Error (MSE) between predicted Q-values and target Q-values
  - Formula: L = (1/B) Σ(Q(s,a) - y)²
  - Where y = r + γ * max Q'(s',a') for non-terminal states
- **Optimizer**: Adam (Adaptive Moment Estimation)
  - Learning rate: α = 0.0001 (with learning rate scheduling)
  - β₁ = 0.9, β₂ = 0.999 (default Adam parameters)
- **Gradient Clipping**: Gradients clipped to [-1, 1] to prevent exploding gradients

### 6.6.3 Target Network

The DQN algorithm employs a **target network** to stabilise training:

```
┌────────────────┐                    ┌────────────────┐
│  Q-Network     │                    │ Target Network │
│  (Online)      │                    │ (θ⁻)           │
│  θ             │                    │                │
├────────────────┤                    ├────────────────┤
│ • Updated      │                    │ • Frozen for   │
│   every step   │                    │   C steps      │
│ • Used for     │     Copy θ → θ⁻   │ • Used for     │
│   action       │◄───every C steps──│   target       │
│   selection    │                    │   calculation  │
│ • Gradient     │                    │ • No gradients │
│   descent      │                    │                │
└────────────────┘                    └────────────────┘
```

**Purpose**: Reduces correlation between target and predicted Q-values, improving training stability.

**Update Frequency**: Every C = 500 steps, parameters are copied from online network to target network.

### 6.6.4 State Representation (Input Features)

The 60-dimensional state vector is constructed as follows:

```python
State Vector [60 dimensions]:
┌─────────────────────────────────────────────────────────────┐
│ [0:10]   Markov Prediction Probabilities                    │
│          Top-10 predicted API calls with their probabilities│
│                                                              │
│ [10:20]  Cache Statistics                                   │
│          • Cache hit rate (current episode)                 │
│          • Cache occupancy ratio                            │
│          • Average response time                            │
│          • Cascade count                                    │
│          • Recent hits (last 5 requests)                    │
│          • Recent misses (last 5 requests)                  │
│                                                              │
│ [20:30]  Request Context                                    │
│          • User type (one-hot encoded: 5 types)            │
│          • Time of day (normalized)                         │
│          • Request rate (requests/second)                   │
│                                                              │
│ [30:50]  API Call History                                   │
│          • Last 20 API calls (one-hot or embedded)         │
│                                                              │
│ [50:60]  System Metrics                                     │
│          • Memory usage                                     │
│          • CPU utilization (if available)                   │
│          • Network latency statistics                       │
│          • Error rate                                       │
└─────────────────────────────────────────────────────────────┘
```

### 6.6.5 Training Hyperparameters

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Learning Rate (α) | 0.0001 | Small value prevents overshooting, suitable for Adam optimizer |
| Discount Factor (γ) | 0.99 | High value emphasizes long-term rewards, suitable for episodic tasks |
| Epsilon Start (ε₀) | 1.0 | Begin with full exploration |
| Epsilon End (ε_min) | 0.01 | Maintain minimal exploration even after convergence |
| Epsilon Decay | 0.995 | Gradual decay allows sufficient exploration |
| Replay Buffer Size | 100,000 | Large enough to decorrelate samples, small enough for memory |
| Batch Size (B) | 64 | Standard batch size balancing variance and computation |
| Target Network Update (C) | 500 steps | Balances stability and adaptability |
| Hidden Dimensions | [256, 256, 128] | Sufficient capacity for complex state-action relationships |

---

## 6.7 Chapter Summary

This chapter has presented a comprehensive design for the Markov Chain-based Reinforcement Learning framework for adaptive API caching in microservices. The design phase has systematically transformed the requirements identified in the SRS into a concrete architectural blueprint that guides implementation.

### Key Design Achievements

**Architectural Foundation**: The four-tier layered architecture (Presentation, Integration/Control, Business Logic/ML, Data/Persistence) provides clear separation of concerns, enabling independent development, testing, and scaling of components. This modular structure facilitates maintenance and future enhancements.

**Design Goals Alignment**: Six primary design goals were established—Performance, Scalability, Accuracy, Maintainability, Reliability, and Extensibility—each with specific measurement criteria. These goals directly map to the non-functional requirements and will be validated during implementation and evaluation.

**Detailed Component Design**: Class diagrams, sequence diagrams, and activity diagrams specify the internal structure and interactions of key subsystems. The Cache Management subsystem employs interface-based design enabling multiple backend implementations. The RL subsystem implements the standard DQN architecture with experience replay and target networks.

**Algorithm Specifications**: Pseudocode for the DQN training algorithm, Markov chain prediction, and action execution provides unambiguous specifications for implementation. The algorithms incorporate established best practices such as double DQN, epsilon-greedy exploration, and experience replay.

**Neural Network Architecture**: The Q-Network design employs a three-hidden-layer fully connected architecture with 60-dimensional input (capturing cache state, predictions, context, and history) and 7-dimensional output (discrete action space). Regularisation techniques (dropout, gradient clipping) and training strategies (target network, Adam optimisation) are incorporated to ensure stable learning.

### Design Rationale

The design decisions documented in this chapter reflect careful consideration of:

1. **Complexity Management**: The layered architecture and modular design manage system complexity by decomposing the problem into manageable components with well-defined interfaces.

2. **ML Integration**: The seamless integration of Markov predictors and DQN agents with traditional caching infrastructure demonstrates how classical and modern AI techniques can complement each other.

3. **Production Readiness**: Design choices such as abstract cache backends, configurable parameters, comprehensive logging, and fault tolerance mechanisms ensure the system can transition from research prototype to production deployment.

4. **Evaluation Facilitation**: The Gymnasium-compliant environment design enables rigorous experimental evaluation using standard RL benchmarking methodologies.

### Transition to Implementation

The design presented in this chapter provides a complete specification for the implementation phase. Each component has clearly defined responsibilities, interfaces, and behaviours. The next chapter (Chapter 7: Implementation) will detail how these designs were realised using specific technologies, frameworks, and libraries, along with the challenges encountered and solutions developed during the implementation process.

The design artefacts—architecture diagrams, class diagrams, sequence diagrams, and algorithm specifications—serve as a contract between design and implementation, enabling systematic validation that the implemented system satisfies the design intent and, ultimately, the system requirements.

---

**End of Chapter 6**
