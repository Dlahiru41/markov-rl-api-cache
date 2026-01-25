"""
Demo script showing practical usage of replay buffers.

This demonstrates how to use both ReplayBuffer and PrioritizedReplayBuffer
in a realistic training scenario.
"""

import numpy as np
from src.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


def demo_basic_replay_buffer():
    """Demonstrate basic ReplayBuffer usage."""
    print("\n" + "="*70)
    print("DEMO: Basic ReplayBuffer (Uniform Sampling)")
    print("="*70)

    # Create buffer
    buffer = ReplayBuffer(capacity=1000, seed=42)
    print(f"\n1. Created buffer with capacity {buffer.capacity}")

    # Simulate collecting experiences
    print("\n2. Collecting experiences from environment...")
    state_dim = 60  # From StateConfig
    num_actions = 7  # From CacheAction

    for episode in range(5):
        print(f"   Episode {episode + 1}:")
        for step in range(20):
            # Simulate environment interaction
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randint(num_actions)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = (step == 19)  # Last step of episode

            # Store in buffer
            buffer.push(state, action, reward, next_state, done)

        print(f"      Buffer size: {len(buffer)}")

    # Training simulation
    print(f"\n3. Training with mini-batches...")
    batch_size = 32

    if buffer.is_ready(batch_size):
        # Sample batch
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        print(f"   Sampled batch of {batch_size}")
        print(f"   - States: {states.shape}, dtype: {states.dtype}")
        print(f"   - Actions: {actions.shape}, dtype: {actions.dtype}")
        print(f"   - Rewards: mean={rewards.mean():.3f}, std={rewards.std():.3f}")
        print(f"   - Dones: {dones.sum():.0f} terminal states")

        # Simulate training step
        print(f"\n   Simulating training step...")
        print(f"   → Compute Q-values")
        print(f"   → Calculate loss")
        print(f"   → Update network")
        print(f"   [OK] Training step complete")

    # Save buffer
    print(f"\n4. Saving buffer for checkpointing...")
    buffer.save("checkpoint_buffer.pkl")
    print(f"   [OK] Saved to checkpoint_buffer.pkl")

    # Clean up
    import os
    os.remove("checkpoint_buffer.pkl")

    print(f"\n[OK] Basic ReplayBuffer demo complete!")


def demo_prioritized_replay_buffer():
    """Demonstrate PrioritizedReplayBuffer usage."""
    print("\n" + "="*70)
    print("DEMO: PrioritizedReplayBuffer")
    print("="*70)

    # Create prioritized buffer
    pbuffer = PrioritizedReplayBuffer(
        capacity=1000,
        alpha=0.6,           # Moderate prioritization
        beta_start=0.4,      # Start with lower IS correction
        beta_end=1.0,        # End with full correction
        beta_frames=10000,   # Anneal over 10k frames
        seed=42
    )

    print(f"\n1. Created prioritized buffer")
    print(f"   - Capacity: {pbuffer.capacity}")
    print(f"   - Alpha (prioritization): {pbuffer.alpha}")
    print(f"   - Beta (IS correction): {pbuffer.beta:.3f} → {pbuffer.beta_end}")

    # Collect experiences
    print(f"\n2. Collecting experiences...")
    state_dim = 60
    num_actions = 7

    for episode in range(5):
        print(f"   Episode {episode + 1}:")
        for step in range(20):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randint(num_actions)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = (step == 19)

            # Store with default priority (will use max_priority)
            pbuffer.push(state, action, reward, next_state, done)

        print(f"      Buffer size: {len(pbuffer)}")

    # Training with priority updates
    print(f"\n3. Training with prioritized sampling...")
    batch_size = 32

    if pbuffer.is_ready(batch_size):
        # Sample prioritized batch
        states, actions, rewards, next_states, dones, weights, indices = \
            pbuffer.sample(batch_size)

        print(f"   Sampled prioritized batch of {batch_size}")
        print(f"   - States: {states.shape}")
        print(f"   - Importance weights: min={weights.min():.3f}, max={weights.max():.3f}")
        print(f"   - Beta (current): {pbuffer.beta:.4f}")

        # Simulate computing TD errors
        print(f"\n   Simulating training step...")
        print(f"   → Compute Q-values")

        # Simulate TD errors (higher error = more surprising = higher priority)
        td_errors = np.abs(np.random.randn(batch_size)) + 0.1
        high_error_idx = np.argmax(td_errors)

        print(f"   → TD errors computed")
        print(f"      - Mean TD-error: {td_errors.mean():.3f}")
        print(f"      - Max TD-error: {td_errors.max():.3f} (at index {high_error_idx})")

        # Weight loss by importance sampling weights
        print(f"   → Apply importance sampling weights to loss")
        weighted_loss = (weights * td_errors ** 2).mean()
        print(f"      - Weighted loss: {weighted_loss:.3f}")

        # Update priorities in buffer
        pbuffer.update_priorities(indices, td_errors)
        print(f"   → Updated priorities in buffer")
        print(f"      - New max priority: {pbuffer.max_priority:.3f}")

        print(f"   [OK] Training step complete")

        # Show effect of prioritization
        print(f"\n4. Demonstrating priority effect...")
        print(f"   Sampling 100 times to see priority distribution...")

        sample_counts = {}
        for _ in range(100):
            _, _, _, _, _, _, sample_indices = pbuffer.sample(5)
            for idx in sample_indices:
                sample_counts[idx] = sample_counts.get(idx, 0) + 1

        top_sampled = sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   Top 3 most sampled experiences:")
        for idx, count in top_sampled:
            print(f"      - Index {idx}: sampled {count} times")

        print(f"\n   → High TD-error experiences are sampled more frequently!")

    # Show beta annealing
    print(f"\n5. Demonstrating beta annealing...")
    initial_beta = pbuffer.beta
    initial_frames = pbuffer.frame_count

    # Sample many times to advance frame count
    for _ in range(50):
        if pbuffer.is_ready(batch_size):
            pbuffer.sample(batch_size)

    print(f"   - Initial: beta={initial_beta:.4f}, frames={initial_frames}")
    print(f"   - After 50 samples: beta={pbuffer.beta:.4f}, frames={pbuffer.frame_count}")
    print(f"   → Beta gradually increases toward {pbuffer.beta_end}")

    print(f"\n[OK] PrioritizedReplayBuffer demo complete!")


def demo_comparison():
    """Compare uniform vs prioritized sampling."""
    print("\n" + "="*70)
    print("DEMO: Uniform vs Prioritized Comparison")
    print("="*70)

    state_dim = 60
    batch_size = 32

    # Create both buffers
    uniform_buffer = ReplayBuffer(capacity=1000, seed=42)
    priority_buffer = PrioritizedReplayBuffer(capacity=1000, alpha=0.8, seed=42)

    # Add same experiences to both
    print(f"\n1. Adding identical experiences to both buffers...")
    for i in range(100):
        state = np.random.randn(state_dim).astype(np.float32)
        action = np.random.randint(7)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = False

        uniform_buffer.push(state, action, reward, next_state, done)
        priority_buffer.push(state, action, reward, next_state, done)

    print(f"   [OK] Both buffers have {len(uniform_buffer)} experiences")

    # Sample from both
    print(f"\n2. Sampling from both buffers...")

    # Uniform sampling
    states_u, actions_u, rewards_u, _, _ = uniform_buffer.sample(batch_size)
    print(f"\n   Uniform Buffer:")
    print(f"   - All experiences have equal probability")
    print(f"   - Reward distribution in sample: mean={rewards_u.mean():.3f}")

    # Prioritized sampling
    states_p, actions_p, rewards_p, _, _, weights, indices = \
        priority_buffer.sample(batch_size)
    print(f"\n   Prioritized Buffer:")
    print(f"   - Experiences sampled by priority")
    print(f"   - Importance weights: min={weights.min():.3f}, max={weights.max():.3f}")
    print(f"   - Reward distribution in sample: mean={rewards_p.mean():.3f}")

    # Simulate setting different priorities
    print(f"\n3. Setting varied priorities...")
    # Give some experiences high priority (simulate high TD-error)
    priorities = np.ones(batch_size) * 0.1
    priorities[:5] = 5.0  # High priority for first 5
    priority_buffer.update_priorities(indices, priorities)

    print(f"   [OK] Set 5 experiences with high priority (5.0)")
    print(f"   [OK] Set rest with low priority (0.1)")

    # Sample again and count
    print(f"\n4. Sampling 100 times to see effect...")
    high_priority_samples = 0
    total_samples = 0

    for _ in range(100):
        _, _, _, _, _, _, sample_idx = priority_buffer.sample(10)
        for idx in sample_idx:
            total_samples += 1
            if idx in indices[:5]:  # High priority experiences
                high_priority_samples += 1

    percentage = (high_priority_samples / total_samples) * 100
    print(f"   - High priority experiences: {high_priority_samples}/{total_samples} ({percentage:.1f}%)")
    print(f"   - Expected if uniform: ~5%")
    print(f"   - Actual with prioritization: {percentage:.1f}%")
    print(f"\n   → Prioritization works! High-priority experiences sampled much more!")

    print(f"\n[OK] Comparison demo complete!")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("REPLAY BUFFER DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows practical usage of replay buffers for DQN training.")

    try:
        # Run demos
        demo_basic_replay_buffer()
        demo_prioritized_replay_buffer()
        demo_comparison()

        print("\n" + "="*70)
        print("[OK] ALL DEMOS COMPLETE")
        print("="*70)
        print("\nKey Takeaways:")
        print("1. ReplayBuffer: Simple, uniform sampling, good baseline")
        print("2. PrioritizedReplayBuffer: Focuses on surprising experiences")
        print("3. Importance sampling: Corrects bias from prioritization")
        print("4. Beta annealing: Starts fast, ends unbiased")
        print("5. Both are production-ready and fully integrated!")

    except Exception as e:
        print(f"\n[FAIL] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

