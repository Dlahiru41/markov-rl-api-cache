"""Comprehensive test suite for the action space module."""
import numpy as np
from src.rl.actions import CacheAction, ActionSpace, ActionConfig, ActionHistory


def test_cache_action_enum():
    """Test CacheAction enum functionality."""
    print("="*70)
    print("Test 1: CacheAction Enum")
    print("="*70)

    # Test num_actions
    assert CacheAction.num_actions() == 7, "Should have 7 actions"
    print("[OK] num_actions() returns 7")

    # Test all action names
    expected_names = [
        "DO_NOTHING", "CACHE_CURRENT", "PREFETCH_CONSERVATIVE",
        "PREFETCH_MODERATE", "PREFETCH_AGGRESSIVE", "EVICT_LRU", "EVICT_LOW_PROB"
    ]
    for i, expected_name in enumerate(expected_names):
        name = CacheAction.get_name(i)
        assert name == expected_name, f"Action {i} should be named {expected_name}, got {name}"
    print("[OK] All action names correct")

    # Test all descriptions exist
    for i in range(7):
        desc = CacheAction.get_description(i)
        assert len(desc) > 0, f"Action {i} should have a description"
        assert "Unknown" not in desc, f"Action {i} has unknown description"
    print("[OK] All actions have descriptions")

    # Test IntEnum properties
    assert CacheAction.DO_NOTHING == 0
    assert CacheAction.EVICT_LOW_PROB == 6
    print("[OK] IntEnum values correct")

    print()


def test_action_config():
    """Test ActionConfig dataclass."""
    print("="*70)
    print("Test 2: ActionConfig")
    print("="*70)

    # Test default config
    config = ActionConfig()
    assert config.conservative_threshold == 0.7
    assert config.moderate_threshold == 0.5
    assert config.aggressive_threshold == 0.3
    assert config.conservative_count == 1
    assert config.moderate_count == 3
    assert config.aggressive_count == 5
    assert config.eviction_batch_size == 10
    print("[OK] Default config values correct")

    # Test custom config
    custom = ActionConfig(
        conservative_threshold=0.9,
        moderate_threshold=0.7,
        aggressive_threshold=0.5,
        conservative_count=2,
        moderate_count=4,
        aggressive_count=6,
        eviction_batch_size=20
    )
    assert custom.conservative_threshold == 0.9
    assert custom.eviction_batch_size == 20
    print("[OK] Custom config works")

    print()


def test_action_space_basic():
    """Test ActionSpace basic functionality."""
    print("="*70)
    print("Test 3: ActionSpace Basic")
    print("="*70)

    space = ActionSpace()

    # Test n property
    assert space.n == 7, "Space should have 7 actions"
    print("[OK] n property correct")

    # Test sample
    for _ in range(20):
        action = space.sample()
        assert 0 <= action < 7, f"Sampled action {action} out of range"
    print("[OK] sample() returns valid actions")

    # Test sampling distribution (rough check)
    samples = [space.sample() for _ in range(1000)]
    unique_actions = set(samples)
    assert len(unique_actions) >= 5, "Should sample diverse actions"
    print("[OK] sample() produces diverse actions")

    print()


def test_valid_actions():
    """Test valid action determination."""
    print("="*70)
    print("Test 4: Valid Actions")
    print("="*70)

    space = ActionSpace()

    # Full cache with predictions
    valid = space.get_valid_actions(
        cache_utilization=0.8,
        has_predictions=True,
        cache_size=100
    )
    assert len(valid) == 7, "All actions should be valid"
    print("[OK] All actions valid when cache has entries and predictions available")

    # No predictions
    valid = space.get_valid_actions(
        cache_utilization=0.5,
        has_predictions=False,
        cache_size=50
    )
    assert CacheAction.DO_NOTHING in valid
    assert CacheAction.CACHE_CURRENT in valid
    assert CacheAction.EVICT_LRU in valid
    assert CacheAction.PREFETCH_CONSERVATIVE not in valid
    assert CacheAction.PREFETCH_MODERATE not in valid
    assert CacheAction.PREFETCH_AGGRESSIVE not in valid
    assert CacheAction.EVICT_LOW_PROB not in valid
    print("[OK] Prefetch actions disabled without predictions")

    # Empty cache
    valid = space.get_valid_actions(
        cache_utilization=0.0,
        has_predictions=True,
        cache_size=0
    )
    assert CacheAction.DO_NOTHING in valid
    assert CacheAction.CACHE_CURRENT in valid
    assert CacheAction.EVICT_LRU not in valid
    assert CacheAction.EVICT_LOW_PROB not in valid
    print("[OK] Eviction actions disabled with empty cache")

    # Minimal valid actions (no predictions, empty cache)
    valid = space.get_valid_actions(
        cache_utilization=0.0,
        has_predictions=False,
        cache_size=0
    )
    assert len(valid) == 2
    assert CacheAction.DO_NOTHING in valid
    assert CacheAction.CACHE_CURRENT in valid
    print("[OK] Only DO_NOTHING and CACHE_CURRENT valid with no predictions and empty cache")

    print()


def test_action_mask():
    """Test action mask generation."""
    print("="*70)
    print("Test 5: Action Mask")
    print("="*70)

    space = ActionSpace()

    # Test mask with all valid
    mask = space.get_action_mask(
        cache_utilization=0.5,
        has_predictions=True,
        cache_size=50
    )
    assert mask.shape == (7,), "Mask should have 7 elements"
    assert mask.dtype == bool, "Mask should be boolean"
    assert mask.all(), "All actions should be valid"
    print("[OK] Mask correct for all valid actions")

    # Test mask with some invalid
    mask = space.get_action_mask(
        cache_utilization=0.0,
        has_predictions=False,
        cache_size=0
    )
    assert mask[0] == True  # DO_NOTHING
    assert mask[1] == True  # CACHE_CURRENT
    assert mask[2] == False  # PREFETCH_CONSERVATIVE
    assert mask[5] == False  # EVICT_LRU
    print("[OK] Mask correct for restricted actions")

    # Test mask matches get_valid_actions
    valid_actions = space.get_valid_actions(0.3, True, 20)
    mask = space.get_action_mask(0.3, True, 20)
    mask_indices = [i for i, v in enumerate(mask) if v]
    assert set(valid_actions) == set(mask_indices), "Mask should match valid actions"
    print("[OK] Mask matches get_valid_actions()")

    print()


def test_decode_action():
    """Test action decoding."""
    print("="*70)
    print("Test 6: Action Decoding")
    print("="*70)

    space = ActionSpace()
    predictions = [('api1', 0.85), ('api2', 0.60), ('api3', 0.40), ('api4', 0.20)]

    # Test DO_NOTHING
    decoded = space.decode_action(CacheAction.DO_NOTHING, predictions)
    assert decoded['action_type'] == 'none'
    assert decoded['cache_current'] == False
    assert decoded['apis_to_prefetch'] == []
    print("[OK] DO_NOTHING decoded correctly")

    # Test CACHE_CURRENT
    decoded = space.decode_action(CacheAction.CACHE_CURRENT, predictions)
    assert decoded['action_type'] == 'cache'
    assert decoded['cache_current'] == True
    print("[OK] CACHE_CURRENT decoded correctly")

    # Test PREFETCH_CONSERVATIVE (threshold=0.7, top-1)
    decoded = space.decode_action(CacheAction.PREFETCH_CONSERVATIVE, predictions)
    assert decoded['action_type'] == 'prefetch'
    assert decoded['apis_to_prefetch'] == ['api1'], f"Expected ['api1'], got {decoded['apis_to_prefetch']}"
    print("[OK] PREFETCH_CONSERVATIVE decoded correctly")

    # Test PREFETCH_MODERATE (threshold=0.5, top-3)
    decoded = space.decode_action(CacheAction.PREFETCH_MODERATE, predictions)
    assert decoded['action_type'] == 'prefetch'
    assert decoded['apis_to_prefetch'] == ['api1', 'api2'], f"Expected ['api1', 'api2'], got {decoded['apis_to_prefetch']}"
    print("[OK] PREFETCH_MODERATE decoded correctly")

    # Test PREFETCH_AGGRESSIVE (threshold=0.3, top-5)
    decoded = space.decode_action(CacheAction.PREFETCH_AGGRESSIVE, predictions)
    assert decoded['action_type'] == 'prefetch'
    assert decoded['apis_to_prefetch'] == ['api1', 'api2', 'api3'], f"Expected ['api1', 'api2', 'api3'], got {decoded['apis_to_prefetch']}"
    print("[OK] PREFETCH_AGGRESSIVE decoded correctly")

    # Test EVICT_LRU
    decoded = space.decode_action(CacheAction.EVICT_LRU, predictions)
    assert decoded['action_type'] == 'evict'
    assert decoded['eviction_strategy'] == 'lru'
    assert decoded['eviction_count'] == 10
    print("[OK] EVICT_LRU decoded correctly")

    # Test EVICT_LOW_PROB
    decoded = space.decode_action(CacheAction.EVICT_LOW_PROB, predictions)
    assert decoded['action_type'] == 'evict'
    assert decoded['eviction_strategy'] == 'low_prob'
    assert decoded['eviction_count'] == 10
    print("[OK] EVICT_LOW_PROB decoded correctly")

    # Test with no predictions
    decoded = space.decode_action(CacheAction.PREFETCH_MODERATE, None)
    assert decoded['apis_to_prefetch'] == []
    print("[OK] Decoding works with no predictions")

    print()


def test_action_history():
    """Test ActionHistory class."""
    print("="*70)
    print("Test 7: Action History")
    print("="*70)

    history = ActionHistory()

    # Record some actions
    state = np.array([1.0, 2.0, 3.0])
    history.record(CacheAction.DO_NOTHING, state, reward=0.5)
    history.record(CacheAction.CACHE_CURRENT, state, reward=1.0)
    history.record(CacheAction.DO_NOTHING, state, reward=0.3)
    history.record(CacheAction.PREFETCH_MODERATE, state, reward=0.8)

    assert len(history.history) == 4, "Should have 4 records"
    print("[OK] Records actions correctly")

    # Test action distribution
    dist = history.get_action_distribution()
    assert dist['DO_NOTHING'] == 0.5  # 2 out of 4
    assert dist['CACHE_CURRENT'] == 0.25  # 1 out of 4
    assert dist['PREFETCH_MODERATE'] == 0.25
    print("[OK] Action distribution correct")

    # Test reward by action
    rewards = history.get_reward_by_action()
    assert abs(rewards['DO_NOTHING'] - 0.4) < 0.01  # (0.5 + 0.3) / 2
    assert rewards['CACHE_CURRENT'] == 1.0
    assert rewards['PREFETCH_MODERATE'] == 0.8
    print("[OK] Reward by action correct")

    # Test statistics
    stats = history.get_statistics()
    assert stats['total_actions'] == 4
    assert stats['total_reward'] == 2.6
    assert abs(stats['average_reward'] - 0.65) < 0.01
    print("[OK] Statistics correct")

    # Test recent actions
    recent = history.get_recent_actions(n=2)
    assert len(recent) == 2
    assert recent[-1]['action'] == CacheAction.PREFETCH_MODERATE
    print("[OK] Recent actions retrieval works")

    # Test clear
    history.clear()
    assert len(history.history) == 0
    assert history.get_action_distribution()['DO_NOTHING'] == 0.0
    print("[OK] Clear works")

    print()


def test_custom_config_integration():
    """Test custom config with ActionSpace."""
    print("="*70)
    print("Test 8: Custom Config Integration")
    print("="*70)

    # Create custom config with stricter thresholds
    config = ActionConfig(
        conservative_threshold=0.9,
        moderate_threshold=0.7,
        aggressive_threshold=0.5
    )
    space = ActionSpace(config=config)

    predictions = [('api1', 0.95), ('api2', 0.75), ('api3', 0.55), ('api4', 0.30)]

    # Conservative should only get api1 (>0.9)
    decoded = space.decode_action(CacheAction.PREFETCH_CONSERVATIVE, predictions)
    assert decoded['apis_to_prefetch'] == ['api1']
    print("[OK] Custom conservative threshold works")

    # Moderate should get api1 and api2 (>0.7)
    decoded = space.decode_action(CacheAction.PREFETCH_MODERATE, predictions)
    assert decoded['apis_to_prefetch'] == ['api1', 'api2']
    print("[OK] Custom moderate threshold works")

    # Aggressive should get api1, api2, api3 (>0.5)
    decoded = space.decode_action(CacheAction.PREFETCH_AGGRESSIVE, predictions)
    assert decoded['apis_to_prefetch'] == ['api1', 'api2', 'api3']
    print("[OK] Custom aggressive threshold works")

    print()


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("="*70)
    print("Test 9: Edge Cases")
    print("="*70)

    space = ActionSpace()

    # Empty predictions list
    decoded = space.decode_action(CacheAction.PREFETCH_AGGRESSIVE, [])
    assert decoded['apis_to_prefetch'] == []
    print("[OK] Handles empty predictions")

    # Predictions all below threshold
    low_preds = [('api1', 0.1), ('api2', 0.05)]
    decoded = space.decode_action(CacheAction.PREFETCH_CONSERVATIVE, low_preds)
    assert decoded['apis_to_prefetch'] == []
    print("[OK] Handles predictions below threshold")

    # More predictions than top_k
    many_preds = [(f'api{i}', 0.9 - i*0.05) for i in range(20)]
    decoded = space.decode_action(CacheAction.PREFETCH_MODERATE, many_preds)
    assert len(decoded['apis_to_prefetch']) <= 3
    print("[OK] Respects top_k limit")

    # Invalid action index (should not crash, handled by dict.get)
    name = CacheAction.get_name(99)
    assert "UNKNOWN" in name
    print("[OK] Handles invalid action index")

    print()


if __name__ == "__main__":
    test_cache_action_enum()
    test_action_config()
    test_action_space_basic()
    test_valid_actions()
    test_action_mask()
    test_decode_action()
    test_action_history()
    test_custom_config_integration()
    test_edge_cases()

    print("="*70)
    print("ALL TESTS PASSED! [OK]")
    print("="*70)

