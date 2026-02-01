"""
Architecture diagram for the Gymnasium Caching Environment.

This file generates a visual representation of how the environment
integrates with all components.
"""

ARCHITECTURE = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                        GYMNASIUM CACHING ENVIRONMENT                       ║
║                            (CachingEnv)                                    ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                     │
                                     │ gymnasium.Env interface
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                             │
        ▼                            ▼                             ▼
   ┌─────────┐                  ┌─────────┐                  ┌──────────┐
   │ reset() │                  │  step() │                  │ render() │
   └────┬────┘                  └────┬────┘                  └──────────┘
        │                            │
        │                            │
        ▼                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      EPISODE MANAGEMENT                                 │
│  • Session generation (user types, API sequences)                      │
│  • Termination logic (cascade, step limit, session end)                │
│  • Metrics tracking (rewards, hits, cascades)                          │
└────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                             │
        ▼                            ▼                             ▼
┌───────────────┐          ┌──────────────────┐         ┌─────────────────┐
│ OBSERVATION   │          │  ACTION          │         │  REWARD         │
│ CONSTRUCTION  │          │  EXECUTION       │         │  CALCULATION    │
└───────────────┘          └──────────────────┘         └─────────────────┘
        │                            │                             │
        │                            │                             │
        ▼                            ▼                             ▼
┌───────────────┐          ┌──────────────────┐         ┌─────────────────┐
│ StateBuilder  │          │   ActionSpace    │         │ RewardCalculator│
│ (from rl/)    │          │   (from rl/)     │         │ (from rl/)      │
└───────────────┘          └──────────────────┘         └─────────────────┘
        │                            │                             │
        │                            │                             │
        └────────────────────────────┴─────────────────────────────┘
                                     │
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                             │
        ▼                            ▼                             ▼
┌──────────────────┐      ┌─────────────────┐         ┌──────────────────┐
│ MarkovPredictor  │      │  CacheManager   │         │   Simulator      │
│ (from markov/)   │      │  (from cache/)  │         │   (mock/real)    │
├──────────────────┤      ├─────────────────┤         ├──────────────────┤
│ • predict()      │      │ • get()         │         │ • Generate APIs  │
│ • observe()      │      │ • set()         │         │ • User sessions  │
│ • reset_history()│      │ • evict_lru()   │         │ • System metrics │
│                  │      │ • evict_low_p() │         │ • Cascade detect │
└──────────────────┘      └─────────────────┘         └──────────────────┘


╔═══════════════════════════════════════════════════════════════════════════╗
║                            DATA FLOW                                       ║
╚═══════════════════════════════════════════════════════════════════════════╝

STEP-BY-STEP FLOW:

1. RESET
   ┌─────────────────────────────────────────────────────────────────┐
   │ env.reset()                                                      │
   │   ├─ Start new session (user type, session length)              │
   │   ├─ Reset Markov predictor history                             │
   │   ├─ Generate first API call                                    │
   │   ├─ Build initial observation (StateBuilder)                   │
   │   └─ Return (observation, info)                                 │
   └─────────────────────────────────────────────────────────────────┘

2. STEP
   ┌─────────────────────────────────────────────────────────────────┐
   │ env.step(action)                                                 │
   │   │                                                              │
   │   ├─ Get Markov predictions (MarkovPredictor.predict())         │
   │   │                                                              │
   │   ├─ Decode action (ActionSpace.decode_action())                │
   │   │   • DO_NOTHING / CACHE / PREFETCH / EVICT                   │
   │   │                                                              │
   │   ├─ Execute action on CacheManager                             │
   │   │   • set() / prefetch() / evict_lru() / evict_low_prob()    │
   │   │                                                              │
   │   ├─ Generate next API call (Simulator)                         │
   │   │                                                              │
   │   ├─ Check cache (CacheManager.get())                           │
   │   │   • HIT → fast response, reward +10                         │
   │   │   • MISS → slow response, reward -1, cache response         │
   │   │                                                              │
   │   ├─ Update Markov predictor (observe())                        │
   │   │                                                              │
   │   ├─ Update system metrics (CPU, memory, latency)               │
   │   │                                                              │
   │   ├─ Check cascade conditions                                   │
   │   │   • Risk score > 0.8 → cascade! reward -100                 │
   │   │   • Risk prevented → reward +50                             │
   │   │                                                              │
   │   ├─ Calculate reward (RewardCalculator)                        │
   │   │   • Multi-objective: cache + cascade + prefetch + latency   │
   │   │                                                              │
   │   ├─ Build new observation (StateBuilder)                       │
   │   │   • Markov predictions (10 dims)                            │
   │   │   • Cache metrics (4 dims)                                  │
   │   │   • System metrics (9 dims)                                 │
   │   │   • Context (12 dims)                                       │
   │   │   • Total: 60-dimensional vector                            │
   │   │                                                              │
   │   ├─ Check termination                                          │
   │   │   • Session ended → terminated=True                         │
   │   │   • Cascade occurred → terminated=True                      │
   │   │   • Step limit → truncated=True                             │
   │   │                                                              │
   │   └─ Return (observation, reward, terminated, truncated, info)  │
   └─────────────────────────────────────────────────────────────────┘


╔═══════════════════════════════════════════════════════════════════════════╗
║                      STATE REPRESENTATION (60D)                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

Observation Vector [0.0 - 1.0]:

[0-4]     Markov API Indices (top-5 predictions, normalized)
[5-9]     Markov Probabilities (confidence scores)
[10]      Markov Confidence (max probability)
[11-14]   Cache Metrics (utilization, hit_rate, entries, eviction_rate)
[15-23]   System Metrics (cpu, mem, req_rate, p50/p95/p99, error, conn, queue)
[24-26]   User Context (is_premium, is_free, is_guest)
[27-32]   Temporal Context (hour_sin/cos, day_sin/cos, is_weekend, is_peak)
[33-35]   Session Context (position, duration, call_count)


╔═══════════════════════════════════════════════════════════════════════════╗
║                        ACTION SPACE (Discrete 7)                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

0: DO_NOTHING           → Let LRU handle everything
1: CACHE_CURRENT        → Explicitly cache current response
2: PREFETCH_CONSERVATIVE → Prefetch top-1 if prob > 70%
3: PREFETCH_MODERATE    → Prefetch top-3 if prob > 50%
4: PREFETCH_AGGRESSIVE  → Prefetch top-5 if prob > 30%
5: EVICT_LRU           → Proactively evict least-recently-used
6: EVICT_LOW_PROB      → Evict entries unlikely to be accessed


╔═══════════════════════════════════════════════════════════════════════════╗
║                         REWARD COMPONENTS                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

Component              │ Reward Value        │ Importance
──────────────────────┼────────────────────┼───────────────
Cache Hit              │ +10.0              │ Baseline good
Cache Miss             │ -1.0               │ Small penalty
Cascade Prevented      │ +50.0 (5x hit)     │ Very important
Cascade Occurred       │ -100.0 (100x miss) │ CATASTROPHIC
Prefetch Used          │ +5.0               │ Moderate
Prefetch Wasted        │ -3.0               │ Moderate penalty
Latency Saved          │ +0.1 per ms        │ Incremental
Latency Added          │ -0.2 per ms        │ Asymmetric
Bandwidth Used         │ -0.01 per KB       │ Small cost
Cache Full (>95%)      │ -5.0               │ Pressure penalty

Total Reward: Clipped to [-100, 100]


╔═══════════════════════════════════════════════════════════════════════════╗
║                    INTEGRATION WITH RL LIBRARIES                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────┐
│                   Stable-Baselines3                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ PPO / DQN / A2C / SAC                                     │  │
│  │   ├─ model = PPO("MlpPolicy", env)                       │  │
│  │   ├─ model.learn(total_timesteps=50_000)                 │  │
│  │   └─ action, _ = model.predict(obs)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              │
                              │ Standard Gym API
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    CachingEnv                                   │
│  • observation_space: Box(60,)                                 │
│  • action_space: Discrete(7)                                   │
│  • reset() → obs, info                                         │
│  • step(action) → obs, reward, terminated, truncated, info     │
└────────────────────────────────────────────────────────────────┘
                              │
                              │ Custom Integration
                              ▼
┌────────────────────────────────────────────────────────────────┐
│            Markov RL API Caching System                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Markov     │  │    Cache     │  │  Services    │         │
│  │  Predictor   │  │   Manager    │  │  Simulator   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────────────────────────────────────────────────┘


USAGE PATTERN:

from stable_baselines3 import PPO
from src.integration.gym_environment import CachingEnv, CacheEnvConfig

# 1. Create environment
config = CacheEnvConfig(max_steps_per_episode=200, seed=42)
env = CachingEnv(config)

# 2. Create RL agent
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train
model.learn(total_timesteps=50_000)

# 4. Evaluate
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        print(f"Episode metrics: {env.get_episode_metrics()}")
        break

env.close()
"""

if __name__ == "__main__":
    print(ARCHITECTURE)

