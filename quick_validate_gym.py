"""
Quick validation script for the Gymnasium caching environment.

This is the validation code from the original user request.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.integration.gym_environment import CachingEnv, CacheEnvConfig
import gymnasium as gym

print("="*60)
print("Quick Validation: Gymnasium Caching Environment")
print("="*60)

# Create environment
config = CacheEnvConfig(
    use_real_services=False,  # Use mock for testing
    max_steps_per_episode=100
)
env = CachingEnv(config)

# Verify it's a valid Gymnasium environment
print(f"\nObservation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Test reset
obs, info = env.reset(seed=42)
print(f"\nInitial observation shape: {obs.shape}")
print(f"Reset info: {info}")

# Test step
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"\nAfter step - Reward: {reward:.2f}, Done: {terminated or truncated}")
print(f"Step info keys: {list(info.keys())}")

# Run a full episode
print("\n" + "-"*60)
print("Running full episode with random policy...")
print("-"*60)

obs, _ = env.reset()
total_reward = 0
steps = 0

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    if terminated or truncated:
        break

print(f"\nEpisode finished: {steps} steps, total reward: {total_reward:.2f}")
print(f"Episode metrics:")
metrics = env.get_episode_metrics()
for key, value in metrics.items():
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

# Verify compatibility with Stable-Baselines3
print("\n" + "-"*60)
print("Checking Stable-Baselines3 compatibility...")
print("-"*60)

try:
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    print("✓ Environment passed Stable-Baselines3 checks!")
except ImportError:
    print("⚠ Stable-Baselines3 not installed - skipping compatibility check")
    print("  Install with: pip install stable-baselines3")
except Exception as e:
    print(f"✗ Compatibility check failed: {e}")

env.close()

print("\n" + "="*60)
print("✓ Validation complete!")
print("="*60)

