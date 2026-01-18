"""
Demo script showing practical usage of Q-Networks.

Demonstrates:
1. Standard QNetwork
2. Dueling QNetwork
3. Training integration
4. Different configurations
"""

import torch
import torch.nn.functional as F
from src.rl.networks import (
    QNetwork, DuelingQNetwork, QNetworkConfig, create_network,
    count_parameters, get_model_summary
)


def demo_standard_qnetwork():
    """Demonstrate standard QNetwork."""
    print("\n" + "="*70)
    print("DEMO 1: Standard QNetwork")
    print("="*70)

    # Create network
    config = QNetworkConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[256, 128, 64],
        activation='relu',
        dropout=0.1
    )

    net = QNetwork(config)
    print(f"\n✓ Created QNetwork")
    print(f"  Parameters: {count_parameters(net):,}")

    # Single state
    print(f"\n1. Testing single state...")
    state = torch.randn(60)
    q_values = net(state)
    action = net.get_action(state)

    print(f"   Input: {state.shape}")
    print(f"   Q-values: {q_values.shape}")
    print(f"   Q-values: [{', '.join(f'{q:.3f}' for q in q_values[:3])}...]")
    print(f"   Best action: {action.item()}")

    # Batch of states
    print(f"\n2. Testing batch of states...")
    batch_size = 32
    states = torch.randn(batch_size, 60)
    q_values = net(states)
    actions = net.get_action(states)

    print(f"   Input: {states.shape}")
    print(f"   Q-values: {q_values.shape}")
    print(f"   Actions: {actions.shape}")
    print(f"   Sample actions: {actions[:5].tolist()}")

    # Training mode vs eval mode
    print(f"\n3. Testing training vs eval mode...")
    net.train()
    q_train1 = net(states)
    q_train2 = net(states)
    print(f"   Training mode (dropout): outputs differ = {not torch.allclose(q_train1, q_train2)}")

    net.eval()
    q_eval1 = net(states)
    q_eval2 = net(states)
    print(f"   Eval mode (no dropout): outputs same = {torch.allclose(q_eval1, q_eval2)}")

    print(f"\n✓ Standard QNetwork demo complete!")


def demo_dueling_qnetwork():
    """Demonstrate Dueling QNetwork."""
    print("\n" + "="*70)
    print("DEMO 2: Dueling QNetwork")
    print("="*70)

    # Create dueling network
    config = QNetworkConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[256, 128, 64],
        activation='relu',
        dropout=0.1,
        dueling=True
    )

    net = DuelingQNetwork(config)
    print(f"\n✓ Created DuelingQNetwork")
    print(f"  Parameters: {count_parameters(net):,}")

    # Forward pass
    print(f"\n1. Testing decomposition...")
    states = torch.randn(32, 60)

    q_values = net(states)
    values = net.get_value(states)
    advantages = net.get_advantage(states)

    print(f"   Q-values: {q_values.shape}")
    print(f"   Values: {values.shape}")
    print(f"   Advantages: {advantages.shape}")

    # Verify decomposition
    reconstructed = values + (advantages - advantages.mean(dim=-1, keepdim=True))
    is_close = torch.allclose(q_values, reconstructed, atol=1e-6)

    print(f"\n2. Verifying decomposition formula...")
    print(f"   Q(s,a) = V(s) + A(s,a) - mean(A)")
    print(f"   Decomposition correct: {is_close}")

    # Show value vs advantage
    print(f"\n3. Sample values and advantages...")
    print(f"   State values: [{', '.join(f'{v.item():.3f}' for v in values[:3])}...]")
    print(f"   Advantages (action 0): [{', '.join(f'{a.item():.3f}' for a in advantages[:3, 0])}...]")
    print(f"   Q-values (action 0): [{', '.join(f'{q.item():.3f}' for q in q_values[:3, 0])}...]")

    print(f"\n✓ Dueling QNetwork demo complete!")


def demo_training_step():
    """Demonstrate training step."""
    print("\n" + "="*70)
    print("DEMO 3: Training Step")
    print("="*70)

    # Setup
    config = QNetworkConfig(state_dim=60, action_dim=7, hidden_dims=[128, 64])
    q_network = QNetwork(config)
    target_network = QNetwork(config)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=1e-4)
    gamma = 0.99

    print(f"\n✓ Setup complete")
    print(f"  Q-network: {count_parameters(q_network):,} parameters")
    print(f"  Optimizer: Adam (lr=1e-4)")
    print(f"  Gamma: {gamma}")

    # Simulate batch from replay buffer
    print(f"\n1. Simulating batch from replay buffer...")
    batch_size = 32
    states = torch.randn(batch_size, 60)
    actions = torch.randint(0, 7, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 60)
    dones = torch.zeros(batch_size)

    print(f"   Batch size: {batch_size}")
    print(f"   Sample rewards: [{', '.join(f'{r.item():.2f}' for r in rewards[:5])}...]")

    # Training step
    print(f"\n2. Performing training step...")
    q_network.train()

    # Current Q-values
    current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
    print(f"   Current Q-values: mean={current_q.mean():.3f}, std={current_q.std():.3f}")

    # Target Q-values
    with torch.no_grad():
        next_q = target_network(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * gamma * next_q

    print(f"   Target Q-values: mean={target_q.mean():.3f}, std={target_q.std():.3f}")

    # Loss
    loss = F.mse_loss(current_q, target_q)
    print(f"   Loss: {loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient norm
    total_norm = torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
    print(f"   Gradient norm: {total_norm:.4f}")

    optimizer.step()
    print(f"   ✓ Parameters updated")

    # Check Q-values after update
    with torch.no_grad():
        new_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        q_change = (new_q - current_q).abs().mean()

    print(f"   Q-value change: {q_change:.6f}")

    print(f"\n✓ Training step demo complete!")


def demo_configurations():
    """Demonstrate different configurations."""
    print("\n" + "="*70)
    print("DEMO 4: Different Configurations")
    print("="*70)

    configs = [
        ("Small", QNetworkConfig(60, 7, [64, 32], 'relu', 0.0)),
        ("Medium", QNetworkConfig(60, 7, [128, 64], 'relu', 0.1)),
        ("Large", QNetworkConfig(60, 7, [256, 128, 64], 'relu', 0.1)),
        ("With LayerNorm", QNetworkConfig(60, 7, [128, 64], 'relu', 0.1, use_layer_norm=True)),
        ("Dueling", QNetworkConfig(60, 7, [128, 64], 'relu', 0.1, dueling=True)),
        ("ELU", QNetworkConfig(60, 7, [128, 64], 'elu', 0.1)),
    ]

    state = torch.randn(16, 60)

    print(f"\n{'Name':<20} {'Parameters':>12} {'Output Shape':>15} {'Mean Q':>10}")
    print("-" * 70)

    for name, config in configs:
        net = create_network(config)
        net.eval()

        with torch.no_grad():
            q_values = net(state)

        params = count_parameters(net)
        mean_q = q_values.mean().item()

        print(f"{name:<20} {params:>12,} {str(q_values.shape):>15} {mean_q:>10.3f}")

    print(f"\n✓ Configurations demo complete!")


def demo_comparison():
    """Compare standard vs dueling networks."""
    print("\n" + "="*70)
    print("DEMO 5: Standard vs Dueling Comparison")
    print("="*70)

    # Create both networks
    config_std = QNetworkConfig(60, 7, [256, 128, 64], dueling=False)
    config_duel = QNetworkConfig(60, 7, [256, 128, 64], dueling=True)

    net_std = QNetwork(config_std)
    net_duel = DuelingQNetwork(config_duel)

    # Compare
    print(f"\n{'Metric':<30} {'Standard':>15} {'Dueling':>15}")
    print("-" * 70)

    params_std = count_parameters(net_std)
    params_duel = count_parameters(net_duel)
    print(f"{'Parameters':<30} {params_std:>15,} {params_duel:>15,}")

    # Forward pass timing
    import time
    state = torch.randn(32, 60)

    # Standard
    start = time.time()
    for _ in range(100):
        _ = net_std(state)
    time_std = (time.time() - start) * 10  # ms per call

    # Dueling
    start = time.time()
    for _ in range(100):
        _ = net_duel(state)
    time_duel = (time.time() - start) * 10  # ms per call

    print(f"{'Time per forward (ms)':<30} {time_std:>15.3f} {time_duel:>15.3f}")

    # Output comparison
    with torch.no_grad():
        q_std = net_std(state)
        q_duel = net_duel(state)

    print(f"{'Mean Q-value':<30} {q_std.mean():>15.3f} {q_duel.mean():>15.3f}")
    print(f"{'Std Q-value':<30} {q_std.std():>15.3f} {q_duel.std():>15.3f}")

    print(f"\n✓ Comparison demo complete!")


def main():
    """Run all demos."""
    print("="*70)
    print("Q-NETWORK DEMONSTRATIONS")
    print("="*70)
    print("\nThis demo shows practical usage of Q-Networks for DQN.")

    try:
        demo_standard_qnetwork()
        demo_dueling_qnetwork()
        demo_training_step()
        demo_configurations()
        demo_comparison()

        print("\n" + "="*70)
        print("✓ ALL DEMOS COMPLETE")
        print("="*70)
        print("\nKey Takeaways:")
        print("1. QNetwork: Standard DQN architecture")
        print("2. DuelingQNetwork: Separate value and advantage streams")
        print("3. Both support batched and single-state inputs")
        print("4. Easy integration with training loops")
        print("5. Flexible configuration options")
        print("\nQ-Networks are ready for DQN training!")

    except Exception as e:
        print(f"\n✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

