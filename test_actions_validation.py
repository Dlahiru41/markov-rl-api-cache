"""Validation script for the action space module."""
from src.rl.actions import CacheAction, ActionSpace, ActionConfig

print("="*70)
print("Action Space Validation")
print("="*70)
print()

# Test enum
print("1. Testing CacheAction Enum:")
print("-" * 70)
print(f"Number of actions: {CacheAction.num_actions()}")
print(f"Action 2 name: {CacheAction.get_name(2)}")
print(f"Action 2 description: {CacheAction.get_description(2)}")
print()

# Test all actions
print("All actions:")
for i in range(CacheAction.num_actions()):
    print(f"  {i}: {CacheAction.get_name(i)}")
    print(f"     {CacheAction.get_description(i)}")
print()

# Test space
print("2. Testing ActionSpace:")
print("-" * 70)
space = ActionSpace()
print(f"Space size (n): {space.n}")
print(f"Random action: {space.sample()}")
print(f"Random action name: {CacheAction.get_name(space.sample())}")
print()

# Test valid actions
print("3. Testing Valid Actions:")
print("-" * 70)
valid = space.get_valid_actions(cache_utilization=0.5, has_predictions=True, cache_size=100)
print(f"Valid actions (cache=50%, predictions=True, size=100):")
print(f"  Indices: {valid}")
print(f"  Names: {[CacheAction.get_name(a) for a in valid]}")
print()

# Test with no predictions
valid_no_pred = space.get_valid_actions(cache_utilization=0.5, has_predictions=False, cache_size=100)
print(f"Valid actions (cache=50%, predictions=False, size=100):")
print(f"  Indices: {valid_no_pred}")
print(f"  Names: {[CacheAction.get_name(a) for a in valid_no_pred]}")
print()

# Test with empty cache
valid_empty = space.get_valid_actions(cache_utilization=0.0, has_predictions=True, cache_size=0)
print(f"Valid actions (cache=0%, predictions=True, size=0):")
print(f"  Indices: {valid_empty}")
print(f"  Names: {[CacheAction.get_name(a) for a in valid_empty]}")
print()

# Test action mask
print("4. Testing Action Mask:")
print("-" * 70)
mask = space.get_action_mask(cache_utilization=0.5, has_predictions=True, cache_size=100)
print(f"Action mask: {mask}")
print(f"Valid actions from mask: {[i for i, valid in enumerate(mask) if valid]}")
print()

# Test decoding
print("5. Testing Action Decoding:")
print("-" * 70)
predictions = [('profile', 0.8), ('browse', 0.55), ('cart', 0.35), ('checkout', 0.1)]
print(f"Predictions: {predictions}")
print()

# Test each prefetch action
for action in [CacheAction.PREFETCH_CONSERVATIVE,
               CacheAction.PREFETCH_MODERATE,
               CacheAction.PREFETCH_AGGRESSIVE]:
    decoded = space.decode_action(action, predictions)
    print(f"{CacheAction.get_name(action)}:")
    print(f"  Type: {decoded['action_type']}")
    print(f"  APIs to prefetch: {decoded['apis_to_prefetch']}")
    print()

# Test other actions
print("DO_NOTHING:")
decoded = space.decode_action(CacheAction.DO_NOTHING, predictions)
print(f"  {decoded}")
print()

print("CACHE_CURRENT:")
decoded = space.decode_action(CacheAction.CACHE_CURRENT, predictions)
print(f"  {decoded}")
print()

print("EVICT_LRU:")
decoded = space.decode_action(CacheAction.EVICT_LRU, predictions)
print(f"  {decoded}")
print()

print("EVICT_LOW_PROB:")
decoded = space.decode_action(CacheAction.EVICT_LOW_PROB, predictions)
print(f"  {decoded}")
print()

# Test custom config
print("6. Testing Custom ActionConfig:")
print("-" * 70)
custom_config = ActionConfig(
    conservative_threshold=0.8,
    moderate_threshold=0.6,
    aggressive_threshold=0.4,
    conservative_count=2,
    moderate_count=4,
    aggressive_count=6,
    eviction_batch_size=20
)
custom_space = ActionSpace(config=custom_config)
decoded_custom = custom_space.decode_action(CacheAction.PREFETCH_MODERATE, predictions)
print(f"Custom config PREFETCH_MODERATE:")
print(f"  Threshold: {custom_config.moderate_threshold}")
print(f"  Count: {custom_config.moderate_count}")
print(f"  APIs to prefetch: {decoded_custom['apis_to_prefetch']}")
print()

print("="*70)
print("✓ All validation tests completed successfully!")
print("="*70)

