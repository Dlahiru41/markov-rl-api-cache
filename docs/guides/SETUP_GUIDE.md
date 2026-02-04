# Quick Setup Guide for Gymnasium Caching Environment

## Installation

### Step 1: Install Core Dependencies

```bash
# Install Gymnasium and NumPy
pip install gymnasium numpy

# Optional: Install Stable-Baselines3 for RL training
pip install stable-baselines3

# Optional: Install matplotlib for visualization
pip install matplotlib

# Or install all at once
pip install -r requirements_gym.txt
```

### Step 2: Verify Installation

Run the quick validation script:

```bash
python quick_validate_gym.py
```

Expected output:
```
============================================================
Quick Validation: Gymnasium Caching Environment
============================================================

Observation space: Box(60,)
Action space: Discrete(7)

Initial observation shape: (60,)
Reset info: {...}

After step - Reward: X.XX, Done: False
Step info keys: [...]

------------------------------------------------------------
Running full episode with random policy...
------------------------------------------------------------

Episode finished: XX steps, total reward: X.XX
Episode metrics:
  ...

------------------------------------------------------------
Checking Stable-Baselines3 compatibility...
------------------------------------------------------------
✓ Environment passed Stable-Baselines3 checks!

============================================================
✓ Validation complete!
============================================================
```

### Step 3: Run Comprehensive Tests

```bash
# Run full test suite (7 tests)
python validate_gym_environment.py
```

Expected output:
```
############################################################
# GYMNASIUM CACHING ENVIRONMENT VALIDATION SUITE
############################################################

============================================================
TEST 1: Environment Creation
============================================================
✓ Environment created successfully
  Observation space: Box(60,)
  Action space: Discrete(7)
  State dimension: 60
✓ Test 1 PASSED

[... more tests ...]

############################################################
# TEST SUMMARY
############################################################
✓ PASSED: Environment Creation
✓ PASSED: Reset Functionality
✓ PASSED: Step Execution
✓ PASSED: Full Episode Rollout
✓ PASSED: Stable-Baselines3 Compatibility
✓ PASSED: Render Functionality
✓ PASSED: Custom Configuration

7/7 tests passed

🎉 ALL TESTS PASSED! Environment is ready for training.
```

## Quick Start Examples

### Example 1: Basic Usage

```python
from src.integration.gym_environment import CachingEnv, CacheEnvConfig

# Create environment
config = CacheEnvConfig(
    use_real_services=False,
    max_steps_per_episode=100,
    seed=42
)
env = CachingEnv(config)

# Reset and run
obs, info = env.reset(seed=42)
print(f"Initial state shape: {obs.shape}")

# Take a step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward:.2f}")
print(f"Cache hit: {info['cache_hit']}")

env.close()
```

### Example 2: Run Full Episode

```python
from src.integration.gym_environment import CachingEnv, CacheEnvConfig

config = CacheEnvConfig(max_steps_per_episode=100, seed=42)
env = CachingEnv(config)

obs, _ = env.reset()
total_reward = 0

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        print(f"Episode ended: {info['episode_summary']}")
        break

print(f"Total reward: {total_reward:.2f}")
env.close()
```

### Example 3: Train with Stable-Baselines3

```python
from stable_baselines3 import PPO
from src.integration.gym_environment import CachingEnv, CacheEnvConfig

# Create environment
config = CacheEnvConfig(max_steps_per_episode=200, seed=42)
env = CachingEnv(config)

# Create and train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# Save model
model.save("ppo_caching_agent")

# Evaluate
obs, _ = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Running Provided Scripts

### 1. Compare Baseline Policies

```bash
python compare_baselines.py
```

This evaluates 6 baseline policies:
- Random
- Do Nothing (LRU)
- Always Cache
- Conservative Prefetch
- Moderate Prefetch
- Adaptive Heuristic

Output includes performance comparison table and detailed analysis.

### 2. Train RL Agents

```bash
python train_rl_agents.py
```

This trains three RL algorithms:
- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Network)
- A2C (Advantage Actor-Critic)

Creates `trained_models/` directory with:
- Trained models
- Evaluation logs
- Checkpoints
- TensorBoard logs
- Performance comparison plot

### 3. Use Trained Models

```python
from stable_baselines3 import PPO
from src.integration.gym_environment import CachingEnv, CacheEnvConfig

# Load trained model
model = PPO.load("trained_models/ppo_final")

# Create environment
config = CacheEnvConfig(max_steps_per_episode=200, seed=42)
env = CachingEnv(config)

# Run with trained agent
obs, _ = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Optional: visualize
    env.render(mode='human')
    
    if terminated or truncated:
        metrics = env.get_episode_metrics()
        print(f"Episode metrics: {metrics}")
        break

env.close()
```

## Troubleshooting

### Issue: "No module named 'gymnasium'"
**Solution**: 
```bash
pip install gymnasium
```

### Issue: "No module named 'stable_baselines3'"
**Solution**: 
```bash
pip install stable-baselines3
```

### Issue: Import errors from src modules
**Solution**: Make sure you're running scripts from the project root directory, or the `src/` directory is in your Python path.

### Issue: Observation out of bounds
**Solution**: Check that all state features are properly normalized in `StateBuilder.build_state()`. All values should be in [0, 1].

### Issue: Episode never ends
**Solution**: Verify `max_steps_per_episode` is set appropriately. Default is 1000 steps.

## Configuration Tips

### For Faster Training
```python
config = CacheEnvConfig(
    max_steps_per_episode=100,  # Shorter episodes
    simulator_config=SimulatorConfig(
        num_apis=10,  # Fewer APIs
        session_length_range=(5, 20)  # Shorter sessions
    ),
    log_episode_metrics=False  # Disable logging
)
```

### For Realistic Simulation
```python
config = CacheEnvConfig(
    max_steps_per_episode=500,  # Longer episodes
    simulator_config=SimulatorConfig(
        num_apis=50,  # More APIs
        session_length_range=(20, 100),  # Longer sessions
        cascade_threshold=0.8  # More realistic threshold
    ),
    episode_end_on_cascade=True  # End on cascade
)
```

### For Custom Rewards
```python
from src.rl.reward import RewardConfig

reward_config = RewardConfig(
    cache_hit_reward=20.0,  # Higher reward
    cascade_prevented_reward=100.0,  # Very high reward
    cascade_occurred_penalty=-200.0  # Severe penalty
)

config = CacheEnvConfig(
    reward_config=reward_config,
    max_steps_per_episode=200
)
```

## Next Steps

1. ✅ Run `python quick_validate_gym.py`
2. ✅ Run `python compare_baselines.py`
3. ✅ Run `python train_rl_agents.py`
4. 📊 Analyze results and tune configurations
5. 🚀 Deploy best-performing agent

## Documentation

- **Comprehensive Guide**: `GYM_ENVIRONMENT_README.md`
- **Implementation Summary**: `GYM_ENVIRONMENT_SUMMARY.md`
- **API Reference**: See docstrings in `src/integration/gym_environment.py`

## Support

For issues, questions, or contributions:
1. Check documentation in `GYM_ENVIRONMENT_README.md`
2. Review validation scripts for usage examples
3. Check existing issues or create a new one

## Success Criteria

Your setup is working if:
- ✅ `quick_validate_gym.py` runs without errors
- ✅ Environment passes SB3 `check_env()`
- ✅ You can train and evaluate RL agents
- ✅ Episode metrics are reasonable (hit rate > 0%, no infinite loops)

**Happy training!** 🎉

