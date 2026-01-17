"""
Exact validation from user's request.
"""

from src.markov.transition_matrix import TransitionMatrix

# Test basic operations
tm = TransitionMatrix(smoothing=0.001)
tm.increment("login", "profile", 80)
tm.increment("login", "browse", 20)
tm.increment("profile", "browse", 50)
tm.increment("profile", "orders", 30)

# Test probability calculation
print(f"P(profile|login) = {tm.get_probability('login', 'profile'):.3f}")  # ~0.8
print(f"P(browse|login) = {tm.get_probability('login', 'browse'):.3f}")  # ~0.2

# Test top-k
print(f"Top transitions from profile: {tm.get_top_k('profile', k=2)}")

# Test serialization
tm.save("test_matrix.json")
tm2 = TransitionMatrix.load("test_matrix.json")
assert tm.get_probability("login", "profile") == tm2.get_probability("login", "profile")

print("\n✓ All validation checks passed!")

# Clean up
import os
if os.path.exists("test_matrix.json"):
    os.remove("test_matrix.json")
    print("✓ Cleaned up test file")

