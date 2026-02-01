# Gymnasium Caching Environment

## Overview

`CachingEnv` is a custom Gymnasium (OpenAI Gym) environment that wraps our entire intelligent caching system for reinforcement learning training. It provides a standard RL interface that allows agents to interact with the Markov predictor, cache manager, and microservices simulator through the standardized Gymnasium API.

## Features

✅ **Standard Gymnasium Interface**: Fully compatible with Stable-Baselines3, RLlib, and other RL libraries  
✅ **Multi-objective Rewards**: Balances cache performance, cascade prevention, prefetch efficiency, and latency  
✅ **Realistic Simulation**: Simulates user sessions with different behavior patterns (guest, free, premium)  
✅ **Comprehensive State**: 60-dimensional state vector with Markov predictions, cache metrics, system metrics, and context  
✅ **7 Discrete Actions**: From conservative to aggressive caching strategies  
✅ **Episode Management**: Natural termination on session end or cascade failure  
✅ **Detailed Metrics**: Track cache hit rate, prediction accuracy, cascade events, and more  
✅ **Configurable**: Highly customizable through dataclass configs  

## Installation

```bash
# Required dependencies
pip install gymnasium numpy

# Optional for RL training
pip install stable-baselines3
```

## Quick Start

```python
from src.integration.gym_environment import CachingEnv, CacheEnvConfig

# Create environment
config = CacheEnvConfig(
    use_real_services=False,  # Use mock for testing
    max_steps_per_episode=100,
    seed=42
)
env = CachingEnv(config)

# Standard Gym interface
obs, info = env.reset(seed=42)
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Run episode
for _ in range(100):
    action = env.action_space.sample()  # Random policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        print(f"Episode ended: {info['episode_summary']}")
        break

env.close()
```

## Environment Specification

### Observation Space

**Type**: `Box(60,)` - Continuous 60-dimensional vector normalized to [0, 1]

**Components**:
- **Markov Predictions** (10 dims): Top-5 API indices + probabilities
- **Confidence** (1 dim): Max prediction probability
- **Cache Metrics** (4 dims): Utilization, hit rate, entries, eviction rate
- **System Metrics** (9 dims): CPU, memory, request rate, latency percentiles (p50/p95/p99), error rate, connections, queue depth
- **User Context** (3 dims): Is premium, is free, is guest (one-hot)
- **Temporal Context** (6 dims): Hour (sin/cos), day (sin/cos), is weekend, is peak hour
- **Session Context** (3 dims): Position in session, duration, call count

### Action Space

**Type**: `Discrete(7)` - Integer actions from 0 to 6

| Action | Name | Description |
|--------|------|-------------|
| 0 | DO_NOTHING | Let normal LRU behavior happen |
| 1 | CACHE_CURRENT | Explicitly cache the current API response |
| 2 | PREFETCH_CONSERVATIVE | Prefetch top-1 prediction if prob > 70% |
| 3 | PREFETCH_MODERATE | Prefetch top-3 predictions if prob > 50% |
| 4 | PREFETCH_AGGRESSIVE | Prefetch top-5 predictions if prob > 30% |
| 5 | EVICT_LRU | Proactively evict least-recently-used entries |
| 6 | EVICT_LOW_PROB | Evict entries with low predicted access probability |

### Reward Function

Multi-objective reward combining:

- **Cache Hits**: +10.0 per hit
- **Cache Misses**: -1.0 per miss
- **Cascade Prevention**: +50.0 (5x cache hit)
- **Cascade Occurrence**: -100.0 (catastrophic)
- **Prefetch Used**: +5.0 per prediction that was actually needed
- **Prefetch Wasted**: -3.0 per prediction that wasn't used
- **Latency**: ±0.1 to ±0.2 per ms saved/added (asymmetric)
- **Bandwidth**: -0.01 per KB used for prefetching
- **Cache Full**: -5.0 when utilization > 95%

**Reward is clipped to [-100, 100] for training stability.**

### Episode Termination

**Terminated** (natural ending):
- Session completed (user finished their browsing session)
- Cascade failure occurred (system overload)

**Truncated** (artificial ending):
- Max steps per episode reached (default: 1000)

## Configuration

### CacheEnvConfig

Main environment configuration:

```python
@dataclass
class CacheEnvConfig:
    # Component configurations
    markov_config: Optional[Dict[str, Any]] = None  # Markov predictor config
    cache_config: Optional[CacheManagerConfig] = None  # Cache manager config
    simulator_config: Optional[SimulatorConfig] = None  # Simulator config
    state_config: Optional[StateConfig] = None  # State representation config
    reward_config: Optional[RewardConfig] = None  # Reward function config
    action_config: Optional[ActionConfig] = None  # Action decoding config
    
    # Episode parameters
    max_steps_per_episode: int = 1000  # Max steps before truncation
    use_real_services: bool = False  # Use actual services or mock
    episode_end_on_cascade: bool = True  # End episode on cascade
    normalize_rewards: bool = False  # Apply reward normalization
    
    # Training parameters
    seed: Optional[int] = None  # Random seed
    log_episode_metrics: bool = True  # Log episode summaries
    render_mode: Optional[str] = None  # 'human', 'ansi', or None
```

### SimulatorConfig

Microservices simulator configuration:

```python
@dataclass
class SimulatorConfig:
    num_apis: int = 20  # Number of different API endpoints
    user_types: List[str] = ['guest', 'free', 'premium']  # User types
    session_length_range: Tuple[int, int] = (10, 100)  # Min/max calls per session
    cascade_threshold: float = 0.8  # System load threshold for cascade
    base_latency_ms: float = 50.0  # Base API response time
    cache_hit_latency_ms: float = 5.0  # Cached response time
    error_rate_threshold: float = 0.1  # Error rate indicating problems
    mock_responses: bool = True  # Use synthetic data
```

### Custom Configuration Example

```python
from src.integration.gym_environment import CachingEnv, CacheEnvConfig, SimulatorConfig
from src.rl.state import StateConfig
from src.rl.reward import RewardConfig

# Custom state configuration
state_config = StateConfig(
    markov_top_k=3,  # Only top-3 predictions
    include_system_metrics=False,  # Disable system metrics
    include_temporal_context=True
)

# Custom reward configuration
reward_config = RewardConfig(
    cache_hit_reward=20.0,  # Higher reward for cache hits
    cascade_prevented_reward=100.0,  # Even higher cascade prevention
    cascade_occurred_penalty=-200.0  # More severe cascade penalty
)

# Custom simulator
simulator_config = SimulatorConfig(
    num_apis=10,  # Smaller API vocabulary
    session_length_range=(5, 30),  # Shorter sessions
    cascade_threshold=0.7  # Lower cascade threshold (more sensitive)
)

# Create environment with custom config
config = CacheEnvConfig(
    state_config=state_config,
    reward_config=reward_config,
    simulator_config=simulator_config,
    max_steps_per_episode=200,
    seed=42
)

env = CachingEnv(config)
```

## Usage with RL Libraries

### Stable-Baselines3

```python
from stable_baselines3 import PPO, DQN, A2C
from src.integration.gym_environment import CachingEnv, CacheEnvConfig

# Create environment
config = CacheEnvConfig(max_steps_per_episode=500, seed=42)
env = CachingEnv(config)

# Train with PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# Evaluate
obs, _ = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

### RLlib (Ray)

```python
from ray.rllib.algorithms.ppo import PPOConfig
from src.integration.gym_environment import CachingEnv, CacheEnvConfig

# RLlib configuration
config = (
    PPOConfig()
    .environment(
        env=CachingEnv,
        env_config={
            "config": CacheEnvConfig(max_steps_per_episode=500, seed=42)
        }
    )
    .framework("torch")
    .training(
        lr=0.0003,
        gamma=0.99,
        train_batch_size=4000
    )
)

algo = config.build()
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward={result['episode_reward_mean']:.2f}")

algo.stop()
```

## Episode Metrics

After each episode, you can retrieve comprehensive metrics:

```python
obs, _ = env.reset()
# ... run episode ...

metrics = env.get_episode_metrics()
```

**Available metrics**:
- `episode_number`: Current episode number
- `total_steps`: Steps taken in episode
- `total_reward`: Sum of all rewards
- `average_reward`: Mean reward per step
- `cache_hit_rate`: % of cache hits
- `total_cache_hits`: Number of cache hits
- `total_cache_misses`: Number of cache misses
- `total_prefetch_hits`: Prefetches that were used
- `total_prefetch_wasted`: Prefetches that weren't used
- `prefetch_efficiency`: Ratio of useful to total prefetches
- `prediction_accuracy`: % of correct Markov predictions
- `action_distribution`: Count of each action taken
- `cascade_occurred`: Whether cascade failure happened
- `final_cascade_risk`: Final cascade risk score
- `session_context`: User type, session length, etc.
- `final_system_metrics`: CPU, memory, latency, etc.
- `final_cache_metrics`: Final cache state

## Rendering

Visualize the environment state:

```python
env.render(mode='human')  # Print to console

output = env.render(mode='ansi')  # Get as string
```

Example output:
```
============================================================
Episode 1 - Step 45
============================================================
User: premium
Session: 45/87
Current API: /api/products/list

Cache State:
  Entries: 12
  Hit Rate: 67.5%

System Metrics:
  CPU: 42.3%
  Memory: 51.2%
  P95 Latency: 78.5ms
  Error Rate: 1.2%
  Cascade Risk: 15.3%

Episode Performance:
  Cumulative Reward: 234.50
  Avg Reward: 5.21
  Cache Hits: 30
  Cache Misses: 15

Recent Actions:
  40: PREFETCH_MODERATE
  41: DO_NOTHING
  42: CACHE_CURRENT
  43: PREFETCH_CONSERVATIVE
  44: DO_NOTHING
============================================================
```

## Validation

Run the validation suite to test all functionality:

```bash
# Comprehensive test suite
python validate_gym_environment.py

# Quick validation (user-provided script)
python quick_validate_gym.py
```

## Architecture

The environment integrates multiple components:

```
CachingEnv
├── MarkovPredictor (from src.markov.predictor)
│   └── Predicts next API calls based on history
├── CacheManager (from src.cache.cache_manager)
│   └── Manages cache operations (get, set, evict, prefetch)
├── StateBuilder (from src.rl.state)
│   └── Constructs observation vectors
├── RewardCalculator (from src.rl.reward)
│   └── Computes multi-objective rewards
└── ActionSpace (from src.rl.actions)
    └── Decodes actions into cache operations
```

## Best Practices

### Training

1. **Start with shorter episodes**: Use `max_steps_per_episode=100-200` initially
2. **Tune reward weights**: Adjust `RewardConfig` based on your priorities
3. **Monitor cascade events**: Track how often cascades occur
4. **Use curriculum learning**: Start with easier scenarios (fewer APIs, shorter sessions)
5. **Evaluate on multiple seeds**: Test trained agents with different seeds

### Debugging

1. **Use render()**: Visualize what the agent is doing
2. **Check episode metrics**: Analyze `get_episode_metrics()` after each episode
3. **Log reward breakdown**: Inspect `info['reward_breakdown']` to see reward components
4. **Track prediction accuracy**: Monitor if Markov predictions are good
5. **Watch cascade risk**: High risk indicates system stress

### Performance

1. **Disable logging**: Set `log_episode_metrics=False` for faster training
2. **Use vectorized environments**: Wrap in `DummyVecEnv` or `SubprocVecEnv`
3. **Adjust state dimension**: Disable unused state components to reduce dimensionality
4. **Batch prefetching**: Use aggressive prefetch actions sparingly (expensive)

## Troubleshooting

### "No module named 'gymnasium'"
```bash
pip install gymnasium
```

### "Observation out of bounds"
Check that all state components are properly normalized in `StateBuilder.build_state()`

### "Episode never ends"
Verify `max_steps_per_episode` is set and `_should_end_episode()` logic is correct

### "Rewards too large/small"
Adjust `RewardConfig.clip_min` and `clip_max`, or enable `normalize_rewards=True`

### "Stable-Baselines3 check fails"
Ensure observation space bounds match actual observations returned by `_build_observation()`

## Future Enhancements

- [ ] Support for continuous action spaces (parameterized prefetch counts, eviction ratios)
- [ ] Multi-agent environments (multiple cache layers, distributed caching)
- [ ] Real service integration (actual API calls instead of mocks)
- [ ] Advanced cascade simulation (network failures, database overload)
- [ ] Hierarchical actions (high-level strategy, low-level tactics)
- [ ] Transfer learning across different API vocabularies
- [ ] Reward shaping based on business metrics (revenue, user satisfaction)

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{caching_gym_env,
  title={Gymnasium Caching Environment for RL-based API Caching},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo/markov-rl-api-cache}
}
```

## License

[Your License Here]

## Contact

For questions or issues, please open a GitHub issue or contact [your email].

