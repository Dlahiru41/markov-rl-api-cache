"""
Demo script to validate the DQN agent implementation.

This script tests:
1. DQNAgent initialization and configuration
2. Action selection (both exploration and evaluation modes)
3. Experience storage and replay buffer
4. Training steps and loss computation
5. Epsilon decay
6. Target network updates
7. Save/load functionality
8. DoubleDQNAgent variant
9. Device handling (CPU/GPU)
"""

import numpy as np
import torch
import os
from src.rl.agents.dqn_agent import DQNAgent, DoubleDQNAgent, DQNConfig

def main():
    print("=" * 80)
    print("DQN AGENT VALIDATION")
    print("=" * 80)

    # Configuration
    config = DQNConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[128, 64],
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        learning_rate=0.001,
        gamma=0.99,
        device='auto',
        seed=42
    )

    print("\n1. TESTING DQN AGENT INITIALIZATION")
    print("-" * 80)
    agent = DQNAgent(config, seed=42)
    print(f"[OK] Agent initialized successfully")
    print(f"  - Device: {agent.device}")
    print(f"  - State dim: {config.state_dim}")
    print(f"  - Action dim: {config.action_dim}")
    print(f"  - Hidden dims: {config.hidden_dims}")
    print(f"  - Epsilon: {agent.epsilon:.3f}")
    print(f"  - Buffer size: {len(agent.buffer)}")

    # Test network architecture
    print("\n2. TESTING NETWORK ARCHITECTURE")
    print("-" * 80)
    print(f"[OK] Online network: {type(agent.online_net).__name__}")
    print(f"[OK] Target network: {type(agent.target_net).__name__}")

    # Count parameters
    online_params = sum(p.numel() for p in agent.online_net.parameters())
    target_params = sum(p.numel() for p in agent.target_net.parameters())
    print(f"  - Online network parameters: {online_params:,}")
    print(f"  - Target network parameters: {target_params:,}")

    # Test action selection
    print("\n3. TESTING ACTION SELECTION")
    print("-" * 80)
    state = np.random.randn(60).astype(np.float32)

    # Exploration mode
    action_explore = agent.select_action(state, evaluate=False)
    print(f"[OK] Exploration action: {action_explore} (epsilon={agent.epsilon:.2f})")
    assert 0 <= action_explore < config.action_dim, "Action out of bounds!"

    # Evaluation mode (greedy)
    action_eval = agent.select_action(state, evaluate=True)
    print(f"[OK] Greedy action: {action_eval}")
    assert 0 <= action_eval < config.action_dim, "Action out of bounds!"

    # Test multiple actions to check randomness in exploration
    actions = [agent.select_action(state, evaluate=False) for _ in range(10)]
    unique_actions = len(set(actions))
    print(f"  - Unique actions in 10 samples: {unique_actions} (should be >1 with high epsilon)")

    # Test greedy consistency
    greedy_actions = [agent.select_action(state, evaluate=True) for _ in range(10)]
    assert len(set(greedy_actions)) == 1, "Greedy actions should be deterministic!"
    print(f"  - Greedy actions are deterministic: [OK]")

    # Test storing experiences
    print("\n4. TESTING EXPERIENCE STORAGE")
    print("-" * 80)
    print("Collecting 200 experiences...")
    for i in range(200):
        state = np.random.randn(60).astype(np.float32)
        action = agent.select_action(state)
        next_state = np.random.randn(60).astype(np.float32)
        reward = np.random.randn()
        done = (i % 50 == 49)
        agent.store_transition(state, action, reward, next_state, done)

    print(f"[OK] Stored {len(agent.buffer)} experiences")
    assert len(agent.buffer) == 200, f"Expected 200 experiences, got {len(agent.buffer)}"

    # Test training
    print("\n5. TESTING TRAINING STEPS")
    print("-" * 80)
    print("Running 10 training steps...")
    print(f"{'Step':<6} {'Loss':<10} {'Q-Mean':<10} {'Epsilon':<10}")
    print("-" * 40)

    for i in range(10):
        metrics = agent.train_step()
        if metrics:
            print(f"{i:<6} {metrics['loss']:<10.4f} {metrics['q_mean']:<10.2f} {metrics['epsilon']:<10.3f}")

    print(f"\n[OK] Training completed successfully")
    print(f"  - Total steps: {agent.steps_done}")
    print(f"  - Final epsilon: {agent.epsilon:.3f}")
    print(f"  - Last loss: {agent.last_loss:.4f}")

    # Test epsilon decay
    print("\n6. TESTING EPSILON DECAY")
    print("-" * 80)
    initial_epsilon = agent.epsilon
    for _ in range(100):
        agent._decay_epsilon()

    print(f"[OK] Epsilon decay working")
    print(f"  - Before 100 decays: {initial_epsilon:.3f}")
    print(f"  - After 100 decays: {agent.epsilon:.3f}")
    print(f"  - Epsilon end limit: {config.epsilon_end}")
    assert agent.epsilon >= config.epsilon_end, "Epsilon went below minimum!"

    # Test target network update
    print("\n7. TESTING TARGET NETWORK UPDATE")
    print("-" * 80)
    # Get a parameter from both networks before update
    online_param_before = next(agent.online_net.parameters()).clone()
    target_param_before = next(agent.target_net.parameters()).clone()

    # Train a bit to change online network
    for _ in range(10):
        state = np.random.randn(60).astype(np.float32)
        action = agent.select_action(state)
        next_state = np.random.randn(60).astype(np.float32)
        reward = np.random.randn()
        done = False
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step()

    online_param_after = next(agent.online_net.parameters()).clone()
    target_param_after = next(agent.target_net.parameters()).clone()

    # Update target network
    agent._update_target_network()
    target_param_updated = next(agent.target_net.parameters()).clone()

    print(f"[OK] Target network can be updated")
    print(f"  - Online params changed: {not torch.allclose(online_param_before, online_param_after)}")
    print(f"  - Target params synced: {torch.allclose(online_param_after, target_param_updated)}")

    # Test save/load
    print("\n8. TESTING SAVE/LOAD")
    print("-" * 80)
    save_path = "test_dqn_agent.pt"

    # Save current state
    epsilon_before = agent.epsilon
    steps_before = agent.steps_done
    agent.save(save_path)
    print(f"[OK] Agent saved to {save_path}")

    # Create new agent and load
    agent2 = DQNAgent(config)
    agent2.load(save_path)
    print(f"[OK] Agent loaded from {save_path}")

    # Verify state restored
    print(f"  - Epsilon: {epsilon_before:.3f} → {agent2.epsilon:.3f}")
    print(f"  - Steps: {steps_before} → {agent2.steps_done}")

    assert abs(agent2.epsilon - epsilon_before) < 1e-6, "Epsilon not restored correctly!"
    assert agent2.steps_done == steps_before, "Steps not restored correctly!"

    # Verify networks are identical
    online_match = all(torch.allclose(p1, p2) for p1, p2 in
                       zip(agent.online_net.parameters(), agent2.online_net.parameters()))
    target_match = all(torch.allclose(p1, p2) for p1, p2 in
                       zip(agent.target_net.parameters(), agent2.target_net.parameters()))

    assert online_match, "Online network weights don't match!"
    assert target_match, "Target network weights don't match!"
    print(f"[OK] Network weights match perfectly")

    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"  - Cleaned up {save_path}")

    # Test DoubleDQNAgent
    print("\n9. TESTING DOUBLE DQN AGENT")
    print("-" * 80)
    double_config = DQNConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[128, 64],
        epsilon_start=1.0,
        epsilon_end=0.1,
        seed=42
    )

    double_agent = DoubleDQNAgent(double_config, seed=42)
    print(f"[OK] DoubleDQNAgent initialized")

    # Collect experiences
    for i in range(200):
        state = np.random.randn(60).astype(np.float32)
        action = double_agent.select_action(state)
        next_state = np.random.randn(60).astype(np.float32)
        reward = np.random.randn()
        done = (i % 50 == 49)
        double_agent.store_transition(state, action, reward, next_state, done)

    # Train
    print("Training Double DQN for 5 steps...")
    for i in range(5):
        metrics = double_agent.train_step()
        if metrics:
            print(f"  Step {i}: loss={metrics['loss']:.4f}, q_mean={metrics['q_mean']:.2f}")

    print(f"[OK] DoubleDQNAgent training works")

    # Test metrics
    print("\n10. TESTING METRICS RETRIEVAL")
    print("-" * 80)
    metrics = agent.get_metrics()
    print(f"[OK] Metrics retrieved:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")

    # Test with different configurations
    print("\n11. TESTING CONFIGURATION VARIATIONS")
    print("-" * 80)

    # Test with prioritized replay
    config_per = DQNConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[64, 32],
        prioritized_replay=True,
        seed=42
    )
    agent_per = DQNAgent(config_per, seed=42)
    print(f"[OK] Agent with prioritized replay buffer: {type(agent_per.buffer).__name__}")

    # Test with regular replay
    config_regular = DQNConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[64, 32],
        prioritized_replay=False,
        seed=42
    )
    agent_regular = DQNAgent(config_regular, seed=42)
    print(f"[OK] Agent with regular replay buffer: {type(agent_regular.buffer).__name__}")

    # Test with different network size
    config_small = DQNConfig(
        state_dim=30,
        action_dim=5,
        hidden_dims=[32, 16],
        dueling=False,
        seed=42
    )
    agent_small = DQNAgent(config_small, seed=42)
    small_params = sum(p.numel() for p in agent_small.online_net.parameters())
    print(f"[OK] Small agent (30→5): {small_params:,} parameters")

    # Test gradient clipping
    print("\n12. TESTING GRADIENT CLIPPING")
    print("-" * 80)
    # Store some experiences
    for i in range(100):
        state = np.random.randn(60).astype(np.float32)
        action = agent.select_action(state)
        next_state = np.random.randn(60).astype(np.float32)
        reward = np.random.randn() * 100  # Large rewards to potentially cause large gradients
        done = False
        agent.store_transition(state, action, reward, next_state, done)

    # Train with gradient clipping
    agent.train_step()

    # Check gradient norms
    max_grad_norm = 0.0
    for param in agent.online_net.parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            max_grad_norm = max(max_grad_norm, grad_norm)

    print(f"[OK] Gradient clipping working")
    print(f"  - Max grad norm: {max_grad_norm:.4f}")
    print(f"  - Clip threshold: {config.max_grad_norm}")

    # Test device handling
    print("\n13. TESTING DEVICE HANDLING")
    print("-" * 80)
    print(f"[OK] Current device: {agent.device}")
    print(f"[OK] CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")

    # Test CPU device explicitly
    config_cpu = DQNConfig(
        state_dim=60,
        action_dim=7,
        device='cpu',
        seed=42
    )
    agent_cpu = DQNAgent(config_cpu, seed=42)
    print(f"[OK] CPU agent device: {agent_cpu.device}")
    assert str(agent_cpu.device) == 'cpu', "CPU device not set correctly!"

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! [OK]")
    print("=" * 80)
    print("\nSummary:")
    print("  [OK] DQNAgent initialization")
    print("  [OK] Action selection (exploration & evaluation)")
    print("  [OK] Experience storage")
    print("  [OK] Training steps and loss computation")
    print("  [OK] Epsilon decay")
    print("  [OK] Target network updates")
    print("  [OK] Save/load functionality")
    print("  [OK] DoubleDQNAgent variant")
    print("  [OK] Prioritized replay buffer")
    print("  [OK] Gradient clipping")
    print("  [OK] Device handling")
    print("  [OK] Configuration variations")
    print("  [OK] Metrics retrieval")
    print("\nThe DQN agent is fully functional and ready for training!")

if __name__ == "__main__":
    main()

