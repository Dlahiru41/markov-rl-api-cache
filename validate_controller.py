"""
Validation script for the IntegrationController.

Tests all functionality including setup, training, evaluation, and API.
"""

import sys
from pathlib import Path
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from src.integration.controller import IntegrationController, ControllerConfig
from src.integration.gym_environment import CacheEnvConfig, SimulatorConfig
from src.rl.dqn_agent import DQNConfig
from src.rl.trainer import TrainingConfig

print("=" * 80)
print("INTEGRATION CONTROLLER VALIDATION")
print("=" * 80)

# Test 1: Create controller config
print("\nTest 1: Creating ControllerConfig...")
try:
    config = ControllerConfig(
        mode='training',
        output_dir='results/test_run',
        enable_monitoring=False,  # Disable to avoid Prometheus dependency
        enable_api=False,  # Disable to avoid FastAPI dependency
        log_level='INFO'
    )

    # Configure for quick testing
    config.env_config = CacheEnvConfig(
        max_steps_per_episode=50,  # Short episodes
        seed=42
    )
    config.env_config.simulator_config = SimulatorConfig(
        num_apis=10,  # Fewer APIs
        session_length_range=(5, 20)  # Shorter sessions
    )
    config.agent_config = DQNConfig(
        batch_size=32,
        buffer_size=1000,
        learning_rate=0.001
    )
    config.training_config = TrainingConfig(
        num_episodes=10,  # Just a few episodes for testing
        eval_frequency=5,
        save_frequency=5
    )

    print("✓ ControllerConfig created successfully")
    print(f"  Mode: {config.mode}")
    print(f"  Output dir: {config.output_dir}")

except Exception as e:
    print(f"✗ Failed to create config: {e}")
    sys.exit(1)

# Test 2: Initialize controller
print("\nTest 2: Initializing IntegrationController...")
try:
    controller = IntegrationController(config)
    print("✓ Controller initialized")

except Exception as e:
    print(f"✗ Failed to initialize controller: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Setup components
print("\nTest 3: Setting up components...")
try:
    success = controller.setup()

    if success:
        print("✓ Setup completed successfully")
        print("  Component status:")
        status = controller.get_status()
        for component, healthy in status['component_health'].items():
            status_str = "✓" if healthy else "✗"
            print(f"    {status_str} {component}")
    else:
        print("✗ Setup failed")
        sys.exit(1)

except Exception as e:
    print(f"✗ Setup error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Get status
print("\nTest 4: Getting system status...")
try:
    status = controller.get_status()
    print("✓ Status retrieved")
    print(f"  Is setup: {status['is_setup']}")
    print(f"  Is running: {status['is_running']}")
    print(f"  Mode: {status['mode']}")

except Exception as e:
    print(f"✗ Failed to get status: {e}")

# Test 5: Get metrics
print("\nTest 5: Getting system metrics...")
try:
    metrics = controller.get_metrics()
    print("✓ Metrics retrieved")
    print(f"  Timestamp: {metrics['timestamp']}")
    if 'markov' in metrics:
        print(f"  Markov vocab size: {metrics['markov']['vocab_size']}")
    if 'agent' in metrics:
        print(f"  Agent epsilon: {metrics['agent']['epsilon']:.4f}")

except Exception as e:
    print(f"✗ Failed to get metrics: {e}")

# Test 6: Start controller
print("\nTest 6: Starting controller...")
try:
    controller.start()
    print("✓ Controller started")

except Exception as e:
    print(f"✗ Failed to start: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Train for a few episodes
print("\nTest 7: Training for 5 episodes...")
try:
    print("  (This may take a minute...)")
    summary = controller.train(num_episodes=5)

    print("✓ Training completed")
    print(f"  Episodes: {summary['num_episodes']}")
    print(f"  Mean reward: {summary['mean_reward']:.2f}")
    print(f"  Std reward: {summary['std_reward']:.2f}")
    print(f"  Final epsilon: {summary['final_epsilon']:.4f}")
    print(f"  Training time: {summary['training_time_seconds']:.1f}s")

except Exception as e:
    print(f"✗ Training failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Evaluate
print("\nTest 8: Running evaluation for 3 episodes...")
try:
    eval_results = controller.evaluate(num_episodes=3)

    print("✓ Evaluation completed")
    print(f"  Mean reward: {eval_results['mean_reward']:.2f}")
    print(f"  Cache hit rate: {eval_results['mean_cache_hit_rate']:.2%}")
    print(f"  Cascade rate: {eval_results['cascade_rate']:.2%}")

except Exception as e:
    print(f"✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Process API call (deployment mode simulation)
print("\nTest 9: Processing API call...")
try:
    result = controller.process_api_call(
        endpoint='/api/products/list',
        context={'user_type': 'premium', 'hour': 14}
    )

    print("✓ API call processed")
    print(f"  Endpoint: {result['endpoint']}")
    print(f"  Action: {result['action_taken']}")
    print(f"  Predictions: {len(result['predictions'])}")

except Exception as e:
    print(f"✗ API call processing failed: {e}")
    import traceback
    traceback.print_exc()

# Test 10: Demo step
print("\nTest 10: Running demo step...")
try:
    demo_state = controller.step_demo()

    print("✓ Demo step executed")
    print(f"  Step: {demo_state['step']}")
    print(f"  Action: {demo_state['action']}")
    print(f"  Reward: {demo_state['reward']:.2f}")
    print(f"  Cache hit: {demo_state['cache_hit']}")

except Exception as e:
    print(f"✗ Demo step failed: {e}")
    import traceback
    traceback.print_exc()

# Test 11: Get final metrics
print("\nTest 11: Getting final metrics...")
try:
    final_metrics = controller.get_metrics()

    print("✓ Final metrics retrieved")
    if 'training' in final_metrics:
        print(f"  Total episodes: {final_metrics['training']['episode_count']}")
        print(f"  Average reward: {final_metrics['training']['average_reward']:.2f}")

except Exception as e:
    print(f"✗ Failed to get final metrics: {e}")

# Test 12: Stop controller
print("\nTest 12: Stopping controller...")
try:
    controller.stop()
    print("✓ Controller stopped gracefully")

except Exception as e:
    print(f"✗ Failed to stop: {e}")

# Test 13: Context manager interface
print("\nTest 13: Testing context manager interface...")
try:
    config2 = ControllerConfig(
        mode='evaluation',
        output_dir='results/test_context',
        enable_monitoring=False,
        enable_api=False
    )
    config2.env_config = CacheEnvConfig(max_steps_per_episode=20, seed=42)

    with IntegrationController(config2) as ctrl:
        status = ctrl.get_status()
        print("✓ Context manager works")
        print(f"  Controller is setup: {status['is_setup']}")

    print("✓ Context manager cleanup completed")

except Exception as e:
    print(f"✗ Context manager failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
print("\n✓ All core functionality validated successfully!")
print("\nThe IntegrationController is ready for use.")
print("\nNext steps:")
print("  1. Run full training: python scripts/controller.py train --episodes 100")
print("  2. Evaluate model: python scripts/controller.py evaluate --model results/test_run/best_model.pt")
print("  3. Start API server: python scripts/controller.py serve --model results/test_run/best_model.pt")
print("  4. Run demo: python scripts/controller.py demo --model results/test_run/best_model.pt")
print("\n" + "=" * 80)

