"""Test that imports work correctly from src.rl package."""
from src.rl import CacheAction, ActionSpace, ActionConfig, ActionHistory

print("Testing imports from src.rl package...")
print()

# Test CacheAction
print(f"[OK] CacheAction imported")
print(f"  Number of actions: {CacheAction.num_actions()}")

# Test ActionSpace
space = ActionSpace()
print(f"[OK] ActionSpace imported")
print(f"  Action space size: {space.n}")

# Test ActionConfig
config = ActionConfig()
print(f"[OK] ActionConfig imported")
print(f"  Conservative threshold: {config.conservative_threshold}")

# Test ActionHistory
history = ActionHistory()
print(f"[OK] ActionHistory imported")
print(f"  History initialized: {len(history.history)} records")

print()
print("All imports successful!")

