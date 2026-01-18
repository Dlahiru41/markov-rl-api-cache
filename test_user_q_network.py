"""
User-provided validation code for Q-Networks.
"""

from src.rl.networks.q_network import QNetwork, DuelingQNetwork, QNetworkConfig, create_network
import torch

config = QNetworkConfig(state_dim=60, action_dim=7, hidden_dims=[256, 128, 64], dueling=False)
net = QNetwork(config)
print(net)
print(f"Parameters: {sum(p.numel() for p in net.parameters())}")

# Test forward pass
state = torch.randn(32, 60)  # Batch of 32 states
q_values = net(state)
print(f"Q-values shape: {q_values.shape}")  # Should be (32, 7)

# Test action selection
actions = net.get_action(state)
print(f"Actions shape: {actions.shape}")  # Should be (32,)

# Test dueling network
config_dueling = QNetworkConfig(state_dim=60, action_dim=7, hidden_dims=[256, 128, 64], dueling=True)
dueling_net = DuelingQNetwork(config_dueling)
print(dueling_net)

q_values = dueling_net(state)
values = dueling_net.get_value(state)
advantages = dueling_net.get_advantage(state)
print(f"Values shape: {values.shape}")  # Should be (32, 1)
print(f"Advantages shape: {advantages.shape}")  # Should be (32, 7)

# Test factory
net2 = create_network(config_dueling)
assert isinstance(net2, DuelingQNetwork)

print("\n✓ User validation code executed successfully!")

