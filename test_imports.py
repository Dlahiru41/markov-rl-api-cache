"""Quick test to check imports."""
import sys
print("Python version:", sys.version)
print("\nTrying to import modules...")

try:
    import gymnasium
    print("✓ gymnasium")
except Exception as e:
    print(f"✗ gymnasium: {e}")

try:
    import torch
    print("✓ torch")
except Exception as e:
    print(f"✗ torch: {e}")

try:
    import numpy
    print("✓ numpy")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import sklearn
    print("✓ sklearn")
except Exception as e:
    print(f"✗ sklearn: {e}")

try:
    import pytest
    print("✓ pytest")
except Exception as e:
    print(f"✗ pytest: {e}")

try:
    from src.integration.gym_environment import CachingEnv
    print("✓ CachingEnv")
except Exception as e:
    print(f"✗ CachingEnv: {e}")

print("\nDone!")

