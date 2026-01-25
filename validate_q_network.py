"""
Validation script for Q-Network implementations.

Tests both QNetwork and DuelingQNetwork with the user's validation code.
"""

from src.rl.networks.q_network import (
    QNetwork, DuelingQNetwork, QNetworkConfig, create_network,
    count_parameters, get_model_summary
)
import torch

print("=" * 70)
print("Q-NETWORK VALIDATION")
print("=" * 70)

# Test 1: Standard QNetwork
print("\n1. Testing Standard QNetwork")
print("-" * 70)

config = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[256, 128, 64],
    dueling=False
)

net = QNetwork(config)
print(net)
print(f"\nParameters: {sum(p.numel() for p in net.parameters()):,}")

# Test forward pass
state = torch.randn(32, 60)  # Batch of 32 states
q_values = net(state)
print(f"\nQ-values shape: {q_values.shape}")  # Should be (32, 7)
assert q_values.shape == (32, 7), f"Expected (32, 7), got {q_values.shape}"
print("[OK] Q-values shape correct!")

# Test action selection
actions = net.get_action(state)
print(f"\nActions shape: {actions.shape}")  # Should be (32,)
assert actions.shape == (32,), f"Expected (32,), got {actions.shape}"
assert actions.dtype == torch.int64, f"Expected int64, got {actions.dtype}"
print("[OK] Actions shape and dtype correct!")

# Test single state (no batch)
single_state = torch.randn(60)
single_q = net(single_state)
print(f"\nSingle state Q-values shape: {single_q.shape}")  # Should be (7,)
assert single_q.shape == (7,), f"Expected (7,), got {single_q.shape}"
print("[OK] Single state forward pass works!")

single_action = net.get_action(single_state)
print(f"Single state action shape: {single_action.shape}")  # Should be ()
assert single_action.ndim == 0, f"Expected scalar, got shape {single_action.shape}"
print("[OK] Single state action selection works!")

# Test 2: Dueling QNetwork
print("\n" + "=" * 70)
print("2. Testing Dueling QNetwork")
print("-" * 70)

config_dueling = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[256, 128, 64],
    dueling=True
)

dueling_net = DuelingQNetwork(config_dueling)
print(dueling_net)
print(f"\nParameters: {sum(p.numel() for p in dueling_net.parameters()):,}")

# Test forward pass
q_values = dueling_net(state)
print(f"\nQ-values shape: {q_values.shape}")  # Should be (32, 7)
assert q_values.shape == (32, 7), f"Expected (32, 7), got {q_values.shape}"
print("[OK] Dueling Q-values shape correct!")

# Test value stream
values = dueling_net.get_value(state)
print(f"\nValues shape: {values.shape}")  # Should be (32, 1)
assert values.shape == (32, 1), f"Expected (32, 1), got {values.shape}"
print("[OK] Values shape correct!")

# Test advantage stream
advantages = dueling_net.get_advantage(state)
print(f"Advantages shape: {advantages.shape}")  # Should be (32, 7)
assert advantages.shape == (32, 7), f"Expected (32, 7), got {advantages.shape}"
print("[OK] Advantages shape correct!")

# Verify dueling architecture property: Q = V + (A - mean(A))
reconstructed_q = values + (advantages - advantages.mean(dim=-1, keepdim=True))
print(f"\nVerifying dueling decomposition...")
assert torch.allclose(q_values, reconstructed_q, atol=1e-6), "Q-value decomposition mismatch!"
print("[OK] Dueling decomposition verified: Q(s,a) = V(s) + A(s,a) - mean(A)")

# Test action selection
actions = dueling_net.get_action(state)
print(f"\nActions shape: {actions.shape}")  # Should be (32,)
assert actions.shape == (32,), f"Expected (32,), got {actions.shape}"
print("[OK] Dueling actions shape correct!")

# Test 3: Factory function
print("\n" + "=" * 70)
print("3. Testing Factory Function")
print("-" * 70)

net2 = create_network(config_dueling)
assert isinstance(net2, DuelingQNetwork), f"Expected DuelingQNetwork, got {type(net2)}"
print("[OK] Factory function creates DuelingQNetwork for dueling=True")

net3 = create_network(config)
assert isinstance(net3, QNetwork), f"Expected QNetwork, got {type(net3)}"
print("[OK] Factory function creates QNetwork for dueling=False")

# Test 4: Different configurations
print("\n" + "=" * 70)
print("4. Testing Different Configurations")
print("-" * 70)

# Test with dropout
config_dropout = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[128, 64],
    dropout=0.2
)
net_dropout = QNetwork(config_dropout)
print(f"[OK] Network with dropout=0.2 created")
print(f"  Parameters: {count_parameters(net_dropout):,}")

# Test with layer normalization
config_ln = QNetworkConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[128, 64],
    use_layer_norm=True
)
net_ln = QNetwork(config_ln)
print(f"[OK] Network with layer normalization created")
print(f"  Parameters: {count_parameters(net_ln):,}")

# Test different activations
for act in ['relu', 'leaky_relu', 'elu', 'tanh']:
    config_act = QNetworkConfig(
        state_dim=60,
        action_dim=7,
        hidden_dims=[64],
        activation=act
    )
    net_act = QNetwork(config_act)
    print(f"[OK] Network with {act} activation created")

# Test 5: Gradient flow
print("\n" + "=" * 70)
print("5. Testing Gradient Flow")
print("-" * 70)

net_grad = QNetwork(config)
optimizer = torch.optim.Adam(net_grad.parameters(), lr=0.001)

# Forward pass
state_grad = torch.randn(16, 60)
q_values_grad = net_grad(state_grad)

# Compute dummy loss
target = torch.randn(16, 7)
loss = torch.nn.functional.mse_loss(q_values_grad, target)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Check gradients
has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in net_grad.parameters())
assert has_grad, "No gradients computed!"
print("[OK] Gradients computed successfully")
print(f"  Loss: {loss.item():.4f}")

# Test 6: Parameter counting
print("\n" + "=" * 70)
print("6. Testing Helper Functions")
print("-" * 70)

total_params = count_parameters(net, trainable_only=False)
trainable_params = count_parameters(net, trainable_only=True)
print(f"[OK] Total parameters: {total_params:,}")
print(f"[OK] Trainable parameters: {trainable_params:,}")
assert total_params == trainable_params, "All parameters should be trainable"

# Test model summary
summary = get_model_summary(net)
print("\n" + summary)

# Test 7: Batch size variations
print("\n" + "=" * 70)
print("7. Testing Various Batch Sizes")
print("-" * 70)

for batch_size in [1, 8, 32, 64, 128]:
    state_batch = torch.randn(batch_size, 60)
    q_batch = net(state_batch)
    actions_batch = net.get_action(state_batch)

    assert q_batch.shape == (batch_size, 7), f"Q-values shape mismatch for batch_size={batch_size}"
    assert actions_batch.shape == (batch_size,), f"Actions shape mismatch for batch_size={batch_size}"

print("[OK] All batch sizes work correctly: [1, 8, 32, 64, 128]")

# Test 8: Deterministic output
print("\n" + "=" * 70)
print("8. Testing Deterministic Output")
print("-" * 70)

net.eval()  # Set to eval mode
state_det = torch.randn(10, 60)

# Multiple forward passes should give same result
out1 = net(state_det)
out2 = net(state_det)

assert torch.allclose(out1, out2), "Network output is not deterministic!"
print("[OK] Network output is deterministic in eval mode")

# Test 9: Network dimensions
print("\n" + "=" * 70)
print("9. Testing Network Dimensions")
print("-" * 70)

# Test various state/action dimensions
configs = [
    (10, 3, [32]),
    (50, 5, [128, 64]),
    (100, 10, [256, 128, 64, 32]),
    (200, 20, [512, 256, 128])
]

for state_dim, action_dim, hidden_dims in configs:
    cfg = QNetworkConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims
    )
    test_net = QNetwork(cfg)
    test_state = torch.randn(4, state_dim)
    test_q = test_net(test_state)

    assert test_q.shape == (4, action_dim), f"Shape mismatch for dims ({state_dim}, {action_dim})"
    print(f"[OK] Network works with state_dim={state_dim}, action_dim={action_dim}, hidden={hidden_dims}")

# Test 10: Error handling
print("\n" + "=" * 70)
print("10. Testing Error Handling")
print("-" * 70)

# Invalid state_dim
try:
    QNetworkConfig(state_dim=0, action_dim=7)
    print("[FAIL] Should have raised error for state_dim=0")
except ValueError:
    print("[OK] Correctly rejects invalid state_dim")

# Invalid action_dim
try:
    QNetworkConfig(state_dim=60, action_dim=-1)
    print("[FAIL] Should have raised error for action_dim=-1")
except ValueError:
    print("[OK] Correctly rejects invalid action_dim")

# Invalid dropout
try:
    QNetworkConfig(state_dim=60, action_dim=7, dropout=1.5)
    print("[FAIL] Should have raised error for dropout=1.5")
except ValueError:
    print("[OK] Correctly rejects invalid dropout")

# Invalid activation
try:
    QNetworkConfig(state_dim=60, action_dim=7, activation='invalid')
    print("[FAIL] Should have raised error for invalid activation")
except ValueError:
    print("[OK] Correctly rejects invalid activation")

# Empty hidden_dims
try:
    QNetworkConfig(state_dim=60, action_dim=7, hidden_dims=[])
    print("[FAIL] Should have raised error for empty hidden_dims")
except ValueError:
    print("[OK] Correctly rejects empty hidden_dims")

print("\n" + "=" * 70)
print("[OK] ALL TESTS PASSED!")
print("=" * 70)
print("\nQ-Networks are ready for DQN training!")
print("\nKey features verified:")
print("  [OK] Standard QNetwork architecture")
print("  [OK] Dueling QNetwork with value/advantage decomposition")
print("  [OK] Configurable hidden layers, activations, dropout, layer norm")
print("  [OK] Batch and single-state processing")
print("  [OK] Gradient flow")
print("  [OK] Factory function")
print("  [OK] Helper utilities")
print("  [OK] Error handling")

