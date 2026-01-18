"""Comprehensive test for state representation module."""
# -*- coding: utf-8 -*-
from src.rl.state import StateBuilder, StateConfig
import numpy as np

def test_basic_functionality():
    """Test basic state building functionality."""
    print("="*60)
    print("Test 1: Basic Functionality")
    print("="*60)

    config = StateConfig(markov_top_k=5)
    print(f"State dimension: {config.state_dim}")

    builder = StateBuilder(config)
    builder.fit(['login', 'profile', 'browse', 'product', 'cart', 'checkout'])

    state = builder.build_state(
        markov_predictions=[('profile', 0.8), ('browse', 0.15), ('cart', 0.05)],
        cache_metrics={'utilization': 0.6, 'hit_rate': 0.75},
        system_metrics={'cpu': 0.3, 'memory': 0.5, 'p95_latency': 150},
        context={'user_type': 'premium', 'hour': 14, 'day': 2}
    )

    print(f"State shape: {state.shape}")
    print(f"State min/max: {state.min():.2f}, {state.max():.2f}")
    print(f"State dtype: {state.dtype}")
    assert state.shape[0] == config.state_dim, f"Shape mismatch: {state.shape[0]} != {config.state_dim}"
    print("✓ Basic functionality test passed!\n")

def test_missing_inputs():
    """Test handling of missing inputs."""
    print("="*60)
    print("Test 2: Missing Inputs")
    print("="*60)

    config = StateConfig(markov_top_k=3)
    builder = StateBuilder(config)
    builder.fit(['api1', 'api2', 'api3'])

    # Build state with missing inputs
    state = builder.build_state()
    print(f"State with all defaults shape: {state.shape}")
    print(f"State with all defaults: {state}")
    assert state.shape[0] == config.state_dim
    print("✓ Missing inputs test passed!\n")

def test_feature_names():
    """Test feature names generation."""
    print("="*60)
    print("Test 3: Feature Names")
    print("="*60)

    config = StateConfig(markov_top_k=3)
    builder = StateBuilder(config)
    builder.fit(['api1', 'api2'])

    feature_names = builder.get_feature_names()
    print(f"Total features: {len(feature_names)}")
    print(f"Feature names: {feature_names}")
    assert len(feature_names) == config.state_dim
    print("✓ Feature names test passed!\n")

def test_cyclical_encoding():
    """Test cyclical encoding for temporal features."""
    print("="*60)
    print("Test 4: Cyclical Encoding")
    print("="*60)

    config = StateConfig(markov_top_k=1, include_probabilities=False,
                        include_confidence=False, include_cache_metrics=False,
                        include_system_metrics=False, include_user_context=False,
                        include_temporal_context=True, include_session_context=False)

    builder = StateBuilder(config)
    builder.fit(['api1'])

    # Test hour 0 (midnight)
    state_0 = builder.build_state(context={'hour': 0, 'day': 0})
    # Test hour 12 (noon)
    state_12 = builder.build_state(context={'hour': 12, 'day': 0})
    # Test hour 23
    state_23 = builder.build_state(context={'hour': 23, 'day': 0})

    print(f"Hour 0: {state_0}")
    print(f"Hour 12: {state_12}")
    print(f"Hour 23: {state_23}")
    print("✓ Cyclical encoding test passed!\n")

def test_value_ranges():
    """Test that all values are in expected ranges."""
    print("="*60)
    print("Test 5: Value Ranges")
    print("="*60)

    config = StateConfig(markov_top_k=5)
    builder = StateBuilder(config)
    builder.fit(['login', 'profile', 'browse', 'product', 'cart'])

    state = builder.build_state(
        markov_predictions=[('profile', 0.9), ('browse', 0.08)],
        cache_metrics={'utilization': 0.8, 'hit_rate': 0.85, 'entries': 500, 'eviction_rate': 10},
        system_metrics={'cpu': 0.7, 'memory': 0.6, 'request_rate': 1000,
                       'p50_latency': 50, 'p95_latency': 200, 'p99_latency': 500,
                       'error_rate': 0.01, 'connections': 100, 'queue_depth': 20},
        context={'user_type': 'free', 'hour': 18, 'day': 5,
                'session_position': 10, 'session_duration': 300, 'call_count': 15}
    )

    print(f"State shape: {state.shape}")
    print(f"State range: [{state.min():.3f}, {state.max():.3f}]")

    # Most values should be in [-1, 1] range (cyclical encoding can be negative)
    print(f"Values outside [-1, 1]: {np.sum((state < -1) | (state > 1))}")

    feature_names = builder.get_feature_names()
    for i, (name, value) in enumerate(zip(feature_names, state)):
        print(f"{i:2d}. {name:25s}: {value:7.3f}")

    print("✓ Value ranges test passed!\n")

def test_different_user_types():
    """Test one-hot encoding for different user types."""
    print("="*60)
    print("Test 6: User Type Encoding")
    print("="*60)

    config = StateConfig(markov_top_k=1, include_probabilities=False,
                        include_confidence=False, include_cache_metrics=False,
                        include_system_metrics=False, include_user_context=True,
                        include_temporal_context=False, include_session_context=False)

    builder = StateBuilder(config)
    builder.fit(['api1'])

    state_premium = builder.build_state(context={'user_type': 'premium'})
    state_free = builder.build_state(context={'user_type': 'free'})
    state_guest = builder.build_state(context={'user_type': 'guest'})
    state_unknown = builder.build_state(context={'user_type': 'unknown'})

    print(f"Premium user: {state_premium}")
    print(f"Free user: {state_free}")
    print(f"Guest user: {state_guest}")
    print(f"Unknown user: {state_unknown}")
    print("✓ User type encoding test passed!\n")

def test_unfitted_builder():
    """Test that unfitted builder raises error."""
    print("="*60)
    print("Test 7: Unfitted Builder Error")
    print("="*60)

    config = StateConfig(markov_top_k=2)
    builder = StateBuilder(config)

    try:
        state = builder.build_state()
        print("✗ Should have raised ValueError!")
        assert False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}\n")

if __name__ == "__main__":
    test_basic_functionality()
    test_missing_inputs()
    test_feature_names()
    test_cyclical_encoding()
    test_value_ranges()
    test_different_user_types()
    test_unfitted_builder()

    print("="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)

