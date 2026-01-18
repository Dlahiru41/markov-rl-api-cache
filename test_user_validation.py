"""
User's exact validation code from the requirements.
"""

from src.rl.agents.dqn_agent import DQNAgent, DoubleDQNAgent, DQNConfig
import numpy as np

config = DQNConfig(
    state_dim=60,
    action_dim=7,
    hidden_dims=[128, 64],
    epsilon_start=1.0,
    epsilon_end=0.1
)
agent = DQNAgent(config, seed=42)

# Test action selection
state = np.random.randn(60).astype(np.float32)
action = agent.select_action(state)
print(f"Selected action: {action} (epsilon={agent.epsilon:.2f})")

# Collect some experiences
for i in range(200):
    state = np.random.randn(60).astype(np.float32)
    action = agent.select_action(state)
    next_state = np.random.randn(60).astype(np.float32)
    reward = np.random.randn()
    done = (i % 50 == 49)
    agent.store_transition(state, action, reward, next_state, done)

# Train
for i in range(10):
    metrics = agent.train_step()
    if metrics:
        print(f"Step {i}: loss={metrics['loss']:.4f}, q_mean={metrics['q_mean']:.2f}, eps={metrics['epsilon']:.3f}")

# Test save/load
agent.save("test_agent.pt")
agent2 = DQNAgent(config)
agent2.load("test_agent.pt")
print(f"Loaded epsilon: {agent2.epsilon}")

