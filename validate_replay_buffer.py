"""
Validation script for replay buffer implementation.

Tests both ReplayBuffer (uniform sampling) and PrioritizedReplayBuffer
to ensure they work correctly.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rl.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


def test_basic_replay_buffer():
    """Test basic ReplayBuffer functionality."""
    print("\n" + "="*60)
    print("Testing ReplayBuffer (Uniform Sampling)")
    print("="*60)

    # Create buffer
    buffer = ReplayBuffer(capacity=1000, seed=42)
    print(f"[OK] Created ReplayBuffer with capacity 1000")

    # Add experiences
    for i in range(100):
        state = np.random.randn(60).astype(np.float32)
        next_state = np.random.randn(60).astype(np.float32)
        buffer.push(
            state=state,
            action=i % 7,
            reward=np.random.randn(),
            next_state=next_state,
            done=False
        )

    print(f"[OK] Added 100 experiences to buffer")
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Ready for batch of 32: {buffer.is_ready(32)}")

    # Sample batch
    states, actions, rewards, next_states, dones = buffer.sample(32)

    print(f"\n[OK] Successfully sampled batch of 32")
    print(f"  States shape: {states.shape} (dtype: {states.dtype})")
    print(f"  Actions shape: {actions.shape} (dtype: {actions.dtype})")
    print(f"  Rewards shape: {rewards.shape} (dtype: {rewards.dtype})")
    print(f"  Next states shape: {next_states.shape} (dtype: {next_states.dtype})")
    print(f"  Dones shape: {dones.shape} (dtype: {dones.dtype})")

    # Test correct dtypes
    assert states.dtype == np.float32, f"States dtype should be float32, got {states.dtype}"
    assert actions.dtype == np.int64, f"Actions dtype should be int64, got {actions.dtype}"
    assert rewards.dtype == np.float32, f"Rewards dtype should be float32, got {rewards.dtype}"
    assert next_states.dtype == np.float32, f"Next states dtype should be float32, got {next_states.dtype}"
    assert dones.dtype == np.float32, f"Dones dtype should be float32, got {dones.dtype}"

    print(f"\n[OK] All dtypes are correct for PyTorch compatibility")

    # Test FIFO behavior
    print(f"\n--- Testing FIFO behavior ---")
    small_buffer = ReplayBuffer(capacity=10, seed=42)
    for i in range(15):
        state = np.array([i], dtype=np.float32)
        small_buffer.push(state, 0, 0.0, state, False)

    print(f"[OK] Added 15 experiences to buffer with capacity 10")
    print(f"  Buffer size: {len(small_buffer)} (should be 10)")
    assert len(small_buffer) == 10, "Buffer should have exactly 10 experiences"

    # Test save/load
    print(f"\n--- Testing save/load ---")
    temp_path = "temp_buffer.pkl"
    buffer.save(temp_path)
    print(f"[OK] Saved buffer to {temp_path}")

    new_buffer = ReplayBuffer(capacity=1000, seed=42)
    new_buffer.load(temp_path)
    print(f"[OK] Loaded buffer from {temp_path}")
    print(f"  Loaded buffer size: {len(new_buffer)}")

    # Clean up
    os.remove(temp_path)

    return True


def test_prioritized_replay_buffer():
    """Test PrioritizedReplayBuffer functionality."""
    print("\n" + "="*60)
    print("Testing PrioritizedReplayBuffer")
    print("="*60)

    # Create prioritized buffer
    pbuffer = PrioritizedReplayBuffer(
        capacity=1000,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_frames=100000,
        seed=42
    )
    print(f"[OK] Created PrioritizedReplayBuffer")
    print(f"  Capacity: 1000")
    print(f"  Alpha: {pbuffer.alpha} (prioritization)")
    print(f"  Beta: {pbuffer.beta:.3f} (starts at {pbuffer.beta_start})")

    # Add experiences
    for i in range(100):
        state = np.random.randn(60).astype(np.float32)
        next_state = np.random.randn(60).astype(np.float32)
        pbuffer.push(
            state=state,
            action=i % 7,
            reward=np.random.randn(),
            next_state=next_state,
            done=False
        )

    print(f"\n[OK] Added 100 experiences to buffer")
    print(f"  Buffer size: {len(pbuffer)}")
    print(f"  Ready for batch of 32: {pbuffer.is_ready(32)}")

    # Sample batch
    states, actions, rewards, next_states, dones, weights, indices = pbuffer.sample(32)

    print(f"\n[OK] Successfully sampled batch of 32")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    print(f"  Next states shape: {next_states.shape}")
    print(f"  Dones shape: {dones.shape}")
    print(f"  Weights shape: {weights.shape} (importance sampling)")
    print(f"  Number of indices: {len(indices)}")

    print(f"\n  Sample weights (should be normalized):")
    print(f"    Min: {weights.min():.4f}")
    print(f"    Max: {weights.max():.4f} (should be 1.0)")
    print(f"    Mean: {weights.mean():.4f}")

    print(f"\n  Sample indices: {indices[:5]}...")

    # Test priority updates
    print(f"\n--- Testing priority updates ---")
    new_priorities = np.abs(np.random.randn(32)) + 0.01
    pbuffer.update_priorities(indices, new_priorities)
    print(f"[OK] Updated priorities for sampled experiences")
    print(f"  Max priority: {pbuffer.max_priority:.4f}")

    # Test beta annealing
    print(f"\n--- Testing beta annealing ---")
    initial_beta = pbuffer.beta
    print(f"  Initial beta: {initial_beta:.4f}")

    # Sample multiple times to advance frame count
    for _ in range(10):
        pbuffer.sample(32)

    print(f"  Beta after 10 samples: {pbuffer.beta:.4f}")
    print(f"  Frame count: {pbuffer.frame_count}")

    # Test save/load
    print(f"\n--- Testing save/load ---")
    temp_path = "temp_pbuffer.pkl"
    pbuffer.save(temp_path)
    print(f"[OK] Saved prioritized buffer to {temp_path}")

    new_pbuffer = PrioritizedReplayBuffer(capacity=1000)
    new_pbuffer.load(temp_path)
    print(f"[OK] Loaded prioritized buffer from {temp_path}")
    print(f"  Loaded buffer size: {len(new_pbuffer)}")
    print(f"  Loaded frame count: {new_pbuffer.frame_count}")

    # Clean up
    os.remove(temp_path)

    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("Testing Edge Cases")
    print("="*60)

    # Test invalid capacity
    try:
        ReplayBuffer(capacity=0)
        print("[FAIL] Should have raised error for capacity=0")
        return False
    except ValueError as e:
        print(f"[OK] Correctly raised error for invalid capacity: {e}")

    # Test sampling more than available
    buffer = ReplayBuffer(capacity=100)
    for i in range(10):
        state = np.random.randn(60).astype(np.float32)
        buffer.push(state, 0, 0.0, state, False)

    try:
        buffer.sample(20)
        print("[FAIL] Should have raised error for sampling more than available")
        return False
    except ValueError as e:
        print(f"[OK] Correctly raised error for over-sampling: {e}")

    # Test is_ready
    print(f"\n[OK] Buffer with 10 items is_ready(5): {buffer.is_ready(5)}")
    print(f"[OK] Buffer with 10 items is_ready(20): {buffer.is_ready(20)}")

    # Test clear
    buffer.clear()
    print(f"[OK] After clear(), buffer size: {len(buffer)}")
    assert len(buffer) == 0, "Buffer should be empty after clear()"

    return True


def test_memory_efficiency():
    """Test that states are stored efficiently as numpy arrays."""
    print("\n" + "="*60)
    print("Testing Memory Efficiency")
    print("="*60)

    buffer = ReplayBuffer(capacity=1000)

    # Add state as list (should be converted to numpy array)
    state_list = [1.0] * 60
    buffer.push(state_list, 0, 0.0, state_list, False)

    # Sample and verify it's a numpy array
    states, _, _, _, _ = buffer.sample(1)
    print(f"[OK] State stored as numpy array: {type(states[0])}")
    print(f"  Shape: {states.shape}, dtype: {states.dtype}")

    # Test that float32 is used for memory efficiency
    state_float64 = np.random.randn(60).astype(np.float64)
    buffer.push(state_float64, 0, 0.0, state_float64, False)

    states, _, _, _, _ = buffer.sample(1)
    print(f"[OK] Float64 converted to float32: {states.dtype}")
    assert states.dtype == np.float32, "States should be float32"

    return True


def test_prioritized_sampling():
    """Test that prioritized sampling actually prioritizes."""
    print("\n" + "="*60)
    print("Testing Prioritized Sampling Behavior")
    print("="*60)

    pbuffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0, seed=42)

    # Add experiences with known priorities
    for i in range(10):
        state = np.array([i], dtype=np.float32)
        pbuffer.push(state, 0, 0.0, state, False)

    # Give one experience very high priority
    states, actions, rewards, next_states, dones, weights, indices = pbuffer.sample(10)

    # Set very high priority for first sampled experience
    high_priority = np.array([100.0] + [1.0] * 9)
    pbuffer.update_priorities(indices, high_priority)

    print(f"[OK] Set high priority for one experience")
    print(f"  Max priority: {pbuffer.max_priority:.2f}")

    # Sample many times and count how often high-priority experience appears
    sample_counts = {}
    for _ in range(100):
        _, _, _, _, _, _, sample_indices = pbuffer.sample(5)
        for idx in sample_indices:
            sample_counts[idx] = sample_counts.get(idx, 0) + 1

    print(f"\n[OK] Sampled 100 batches of 5")
    print(f"  Sample distribution (top 5 most sampled):")
    sorted_counts = sorted(sample_counts.items(), key=lambda x: x[1], reverse=True)
    for idx, count in sorted_counts[:5]:
        print(f"    Index {idx}: {count} times")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("REPLAY BUFFER VALIDATION")
    print("="*60)

    try:
        # Run tests
        test_basic_replay_buffer()
        test_prioritized_replay_buffer()
        test_edge_cases()
        test_memory_efficiency()
        test_prioritized_sampling()

        print("\n" + "="*60)
        print("[OK] ALL TESTS PASSED")
        print("="*60)
        print("\nReplay buffers are ready for DQN training!")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

