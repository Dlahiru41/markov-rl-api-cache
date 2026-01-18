"""Validation script for the state representation module."""
from src.rl.state import StateBuilder, StateConfig
import numpy as np

# Test configuration
config = StateConfig(markov_top_k=5)
print(f"State dimension: {config.state_dim}")

# Create and fit builder
builder = StateBuilder(config)
builder.fit(['login', 'profile', 'browse', 'product', 'cart', 'checkout'])

# Build state
state = builder.build_state(
    markov_predictions=[('profile', 0.8), ('browse', 0.15), ('cart', 0.05)],
    cache_metrics={'utilization': 0.6, 'hit_rate': 0.75},
    system_metrics={'cpu': 0.3, 'memory': 0.5, 'p95_latency': 150},
    context={'user_type': 'premium', 'hour': 14, 'day': 2}
)

print(f"State shape: {state.shape}")  # Should match config.state_dim
print(f"State min/max: {state.min():.2f}, {state.max():.2f}")  # Should be reasonable range
print(f"Feature names (first 10): {builder.get_feature_names()[:10]}")
print(f"Total features: {len(builder.get_feature_names())}")
print(f"\nState vector:\n{state}")

