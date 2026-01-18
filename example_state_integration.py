"""
Integration example showing how to use StateBuilder with other system components.

This demonstrates a realistic usage scenario where the RL agent uses StateBuilder
to convert raw system information into state vectors for decision-making.
"""

import numpy as np
from src.rl.state import StateBuilder, StateConfig

class MockMarkovPredictor:
    """Simulates a Markov predictor for demonstration."""

    def predict(self, history, top_k=5):
        """Return mock predictions."""
        predictions = [
            ('profile', 0.45),
            ('browse', 0.25),
            ('cart', 0.15),
            ('checkout', 0.10),
            ('payment', 0.05)
        ]
        return predictions[:top_k]

class MockCacheSystem:
    """Simulates a cache system for demonstration."""

    def get_metrics(self):
        """Return mock cache metrics."""
        return {
            'utilization': 0.65,
            'hit_rate': 0.78,
            'entries': 1250,
            'eviction_rate': 25.5
        }

class MockMonitoringSystem:
    """Simulates a monitoring system for demonstration."""

    def get_metrics(self):
        """Return mock system metrics."""
        return {
            'cpu': 0.45,
            'memory': 0.62,
            'request_rate': 1850,
            'p50_latency': 45,
            'p95_latency': 180,
            'p99_latency': 350,
            'error_rate': 0.002,
            'connections': 125,
            'queue_depth': 15
        }

class MockContextProvider:
    """Simulates a context provider for demonstration."""

    def get_context(self):
        """Return mock context information."""
        return {
            'user_type': 'premium',
            'hour': 15,  # 3 PM
            'day': 2,    # Tuesday
            'session_position': 12,
            'session_duration': 450,  # 7.5 minutes
            'call_count': 18
        }

class SimpleCachingAgent:
    """Simple RL agent that uses state vectors to make caching decisions."""

    def __init__(self, state_builder):
        self.state_builder = state_builder
        self.state_dim = state_builder.config.state_dim

    def select_action(self, state):
        """
        Select caching action based on state.

        In a real implementation, this would use a neural network.
        For demonstration, we use a simple heuristic.
        """
        # Get markov confidence (index 10 in default config)
        markov_confidence = state[10] if len(state) > 10 else 0.0

        # Get cache hit rate (index 12 in default config)
        cache_hit_rate = state[12] if len(state) > 12 else 0.0

        # Simple decision rule
        if markov_confidence > 0.7 and cache_hit_rate < 0.9:
            return "cache"  # Cache the predicted API
        elif cache_hit_rate > 0.95:
            return "no_action"  # Cache is performing well
        else:
            return "evict_lru"  # Evict least recently used

    def get_state_analysis(self, state):
        """Analyze the state vector for debugging."""
        feature_names = self.state_builder.get_feature_names()

        analysis = {
            'state_dim': len(state),
            'value_range': (float(state.min()), float(state.max())),
            'top_features': []
        }

        # Find top 5 non-zero features
        non_zero_indices = np.where(np.abs(state) > 0.01)[0]
        top_indices = non_zero_indices[np.argsort(np.abs(state[non_zero_indices]))[-5:]]

        for idx in top_indices:
            analysis['top_features'].append({
                'name': feature_names[idx],
                'value': float(state[idx])
            })

        return analysis

def main():
    """Run integration example."""
    print("="*70)
    print("State Representation Integration Example")
    print("="*70)
    print()

    # 1. Setup components
    print("1. Initializing components...")

    # Create API vocabulary
    api_vocabulary = [
        'login', 'logout', 'profile', 'browse', 'search',
        'product', 'cart', 'checkout', 'payment', 'confirmation',
        'settings', 'help', 'contact', 'about', 'terms'
    ]

    # Configure state builder
    config = StateConfig(markov_top_k=5)
    state_builder = StateBuilder(config)
    state_builder.fit(api_vocabulary)

    # Create mock components
    markov_predictor = MockMarkovPredictor()
    cache_system = MockCacheSystem()
    monitoring = MockMonitoringSystem()
    context_provider = MockContextProvider()

    # Create agent
    agent = SimpleCachingAgent(state_builder)

    print(f"   - API vocabulary size: {len(api_vocabulary)}")
    print(f"   - State dimension: {config.state_dim}")
    print()

    # 2. Simulate decision cycle
    print("2. Running decision cycle...")
    print()

    for cycle in range(3):
        print(f"   Cycle {cycle + 1}:")
        print("   " + "-"*60)

        # Gather information from all components
        predictions = markov_predictor.predict(['login', 'profile'])
        cache_metrics = cache_system.get_metrics()
        system_metrics = monitoring.get_metrics()
        context = context_provider.get_context()

        # Build state vector
        state = state_builder.build_state(
            markov_predictions=predictions,
            cache_metrics=cache_metrics,
            system_metrics=system_metrics,
            context=context
        )

        # Agent makes decision
        action = agent.select_action(state)

        # Analyze state
        analysis = agent.get_state_analysis(state)

        print(f"   State shape: {state.shape}")
        print(f"   State range: [{analysis['value_range'][0]:.3f}, {analysis['value_range'][1]:.3f}]")
        print(f"   Selected action: {action}")
        print()
        print("   Top features:")
        for feat in analysis['top_features']:
            print(f"      - {feat['name']:30s}: {feat['value']:7.3f}")
        print()

    # 3. Feature visualization
    print("3. Feature breakdown (Cycle 1):")
    print("   " + "-"*60)

    # Get fresh state
    state = state_builder.build_state(
        markov_predictions=markov_predictor.predict(['login']),
        cache_metrics=cache_system.get_metrics(),
        system_metrics=monitoring.get_metrics(),
        context=context_provider.get_context()
    )

    feature_names = state_builder.get_feature_names()

    # Group features by category
    categories = {
        'Markov Predictions': range(0, 11),
        'Cache Metrics': range(11, 15),
        'System Metrics': range(15, 24),
        'User Context': range(24, 27),
        'Temporal Context': range(27, 33),
        'Session Context': range(33, 36)
    }

    for category, indices in categories.items():
        print(f"\n   {category}:")
        for idx in indices:
            if idx < len(state):
                print(f"      {feature_names[idx]:30s}: {state[idx]:7.3f}")

    print()
    print("="*70)
    print("Integration example completed successfully!")
    print("="*70)

if __name__ == "__main__":
    main()

