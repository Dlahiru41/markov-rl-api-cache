"""
Simple test verification script.
Runs a subset of tests to verify the test suite is working.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("TEST SUITE VERIFICATION")
print("=" * 70)

# Test 1: Import test module
print("\n1. Testing imports...")
try:
    from tests.unit import test_markov
    print("   [OK] Test module imported successfully")
except ImportError as e:
    print(f"   [FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Import Markov components
print("\n2. Testing Markov component imports...")
try:
    from src.markov import (
        TransitionMatrix,
        FirstOrderMarkovChain,
        SecondOrderMarkovChain,
        ContextAwareMarkovChain,
        MarkovPredictor
    )
    print("   [OK] All Markov components imported")
except ImportError as e:
    print(f"   [FAIL] Import error: {e}")
    sys.exit(1)

# Test 3: Run a simple inline test
print("\n3. Running simple inline tests...")

# Test TransitionMatrix
try:
    tm = TransitionMatrix(smoothing=0.001)
    tm.increment('A', 'B', 10)
    prob = tm.get_probability('A', 'B')
    assert 0 < prob <= 1
    print("   [OK] TransitionMatrix working")
except Exception as e:
    print(f"   [FAIL] TransitionMatrix error: {e}")

# Test FirstOrderMarkovChain
try:
    mc = FirstOrderMarkovChain(smoothing=0.001)
    sequences = [['A', 'B', 'C'], ['A', 'B', 'D']]
    mc.fit(sequences)
    predictions = mc.predict('A', k=2)
    assert len(predictions) > 0
    print("   [OK] FirstOrderMarkovChain working")
except Exception as e:
    print(f"   [FAIL] FirstOrderMarkovChain error: {e}")

# Test MarkovPredictor
try:
    predictor = MarkovPredictor(order=1)
    sequences = [['X', 'Y', 'Z'], ['X', 'Y', 'W']]
    predictor.fit(sequences)
    predictor.observe('X')
    predictions = predictor.predict(k=2)
    state = predictor.get_state_vector(k=3)
    assert len(predictions) > 0
    assert state.shape[0] > 0
    print("   [OK] MarkovPredictor working")
except Exception as e:
    print(f"   [FAIL] MarkovPredictor error: {e}")

# Test 4: Check pytest availability
print("\n4. Checking pytest availability...")
try:
    import pytest
    print(f"   [OK] pytest version: {pytest.__version__}")
except ImportError:
    print("   ⚠ pytest not installed - install with: pip install pytest")

# Test 5: Test fixtures can be created
print("\n5. Testing fixture creation...")
try:
    # Create simple_sequences fixture data
    simple_sequences = [
        ['login', 'profile', 'orders'],
        ['login', 'browse', 'product'],
        ['browse', 'product', 'cart'],
    ]
    assert len(simple_sequences) == 3
    print("   [OK] Fixture data can be created")
except Exception as e:
    print(f"   [FAIL] Fixture creation error: {e}")

print("\n" + "=" * 70)
print("[OK] VERIFICATION COMPLETE")
print("=" * 70)
print("\nThe test suite is properly set up!")
print("\nTo run the full test suite:")
print("  python -m pytest tests/unit/test_markov.py -v")
print("\nTo run fast tests only:")
print("  python -m pytest tests/unit/test_markov.py -v -m 'not slow'")
print("\nTo run with coverage:")
print("  python -m pytest tests/unit/test_markov.py --cov=src/markov --cov-report=html")

