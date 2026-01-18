"""
User-provided validation code for replay buffers.
"""

from src.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import numpy as np

# Test basic buffer
buffer = ReplayBuffer(capacity=1000, seed=42)
for i in range(100):
    state = np.random.randn(60).astype(np.float32)
    next_state = np.random.randn(60).astype(np.float32)
    buffer.push(state, action=i % 7, reward=np.random.randn(),
                next_state=next_state, done=False)

print(f"Buffer size: {len(buffer)}")
print(f"Ready for batch of 32: {buffer.is_ready(32)}")

states, actions, rewards, next_states, dones = buffer.sample(32)
print(f"Batch shapes: states={states.shape}, actions={actions.shape}")

# Test prioritized buffer
pbuffer = PrioritizedReplayBuffer(capacity=1000)
for i in range(100):
    state = np.random.randn(60).astype(np.float32)
    next_state = np.random.randn(60).astype(np.float32)
    pbuffer.push(state, action=i % 7, reward=np.random.randn(),
                 next_state=next_state, done=False)

states, actions, rewards, next_states, dones, weights, indices = pbuffer.sample(32)
print(f"Weights shape: {weights.shape}")
print(f"Indices: {indices[:5]}")

# Update priorities
new_priorities = np.abs(np.random.randn(32)) + 0.01
pbuffer.update_priorities(indices, new_priorities)

print("\n✓ User validation code executed successfully!")

