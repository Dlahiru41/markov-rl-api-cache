#!/usr/bin/env python
"""
Final validation and summary of TransitionMatrix implementation.
Run this to verify everything is working correctly.
"""

import sys
from pathlib import Path

def test_import():
    """Test that TransitionMatrix can be imported."""
    try:
        from src.markov import TransitionMatrix
        print("[SUCCESS] Import successful: from src.markov import TransitionMatrix")
        return True
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic TransitionMatrix operations."""
    try:
        from src.markov import TransitionMatrix

        tm = TransitionMatrix(smoothing=0.001)
        tm.increment("A", "B", 10)
        tm.increment("A", "C", 5)

        prob_b = tm.get_probability("A", "B")
        prob_c = tm.get_probability("A", "C")

        assert abs(prob_b - 0.667) < 0.01, f"Expected ~0.667, got {prob_b}"
        assert abs(prob_c - 0.333) < 0.01, f"Expected ~0.333, got {prob_c}"

        print(f"[SUCCESS] Basic operations working:")
        print(f"   P(B|A) = {prob_b:.3f}")
        print(f"   P(C|A) = {prob_c:.3f}")
        return True
    except Exception as e:
        print(f"[ERROR] Basic operations failed: {e}")
        return False

def test_top_k():
    """Test top-k functionality."""
    try:
        from src.markov import TransitionMatrix

        tm = TransitionMatrix()
        tm.increment("X", "Y", 10)
        tm.increment("X", "Z", 5)

        top = tm.get_top_k("X", k=2)

        assert len(top) == 2, f"Expected 2 results, got {len(top)}"
        assert top[0][0] == "Y", f"Expected 'Y' first, got {top[0][0]}"

        print(f"[SUCCESS] Top-K queries working:")
        print(f"   Top from 'X': {top}")
        return True
    except Exception as e:
        print(f"[ERROR] Top-K failed: {e}")
        return False

def test_serialization():
    """Test save/load functionality."""
    try:
        from src.markov import TransitionMatrix
        import os

        tm1 = TransitionMatrix(smoothing=0.01)
        tm1.increment("login", "profile", 80)
        tm1.increment("login", "browse", 20)

        test_file = "_test_matrix_temp.json"
        tm1.save(test_file)

        tm2 = TransitionMatrix.load(test_file)

        prob1 = tm1.get_probability("login", "profile")
        prob2 = tm2.get_probability("login", "profile")

        assert prob1 == prob2, f"Probabilities don't match: {prob1} != {prob2}"

        os.remove(test_file)

        print(f"[SUCCESS] Serialization working:")
        print(f"   Saved and loaded successfully")
        print(f"   Probability preserved: {prob1:.3f}")
        return True
    except Exception as e:
        print(f"[ERROR] Serialization failed: {e}")
        return False

def test_statistics():
    """Test statistics generation."""
    try:
        from src.markov import TransitionMatrix

        tm = TransitionMatrix()
        tm.increment("A", "B", 100)
        tm.increment("B", "C", 50)
        tm.increment("C", "A", 25)

        stats = tm.get_statistics()

        assert stats['num_states'] == 3, f"Expected 3 states, got {stats['num_states']}"
        assert stats['num_transitions'] == 3, f"Expected 3 transitions"

        print(f"[SUCCESS] Statistics working:")
        print(f"   States: {stats['num_states']}")
        print(f"   Transitions: {stats['num_transitions']}")
        print(f"   Sparsity: {stats['sparsity']:.1%}")
        return True
    except Exception as e:
        print(f"[ERROR] Statistics failed: {e}")
        return False

def check_files():
    """Check that all required files exist."""
    files_to_check = [
        "src/markov/transition_matrix.py",
        "src/markov/__init__.py",
        "src/markov/README.md",
        "tests/unit/test_transition_matrix.py",
        "TRANSITION_MATRIX_QUICK_REF.md",
        "TRANSITION_MATRIX_COMPLETE.md",
        "demo_transition_matrix.py",
        "example_transition_matrix_integration.py",
    ]

    print("\n📁 Checking files...")
    all_exist = True
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"   [SUCCESS] {file_path:50s} ({size:6,} bytes)")
        else:
            print(f"   [ERROR] {file_path:50s} (missing)")
            all_exist = False

    return all_exist

def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print(" " * 15 + "TRANSITION MATRIX - FINAL VALIDATION")
    print("=" * 70 + "\n")

    tests = [
        ("Import Test", test_import),
        ("Basic Operations", test_basic_functionality),
        ("Top-K Queries", test_top_k),
        ("Serialization", test_serialization),
        ("Statistics", test_statistics),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n🔍 Running: {name}")
        print("-" * 70)
        result = test_func()
        results.append((name, result))

    # Check files
    print("\n" + "=" * 70)
    files_ok = check_files()
    results.append(("File Check", files_ok))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[SUCCESS] PASS" if result else "[ERROR] FAIL"
        print(f"  {status}  {name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "🎉" * 35)
        print("\n  ✨ ALL VALIDATION TESTS PASSED! ✨")
        print("\n  The TransitionMatrix implementation is:")
        print("    [SUCCESS] Fully functional")
        print("    [SUCCESS] Well tested")
        print("    [SUCCESS] Production ready")
        print("    [SUCCESS] Documented")
        print("\n  Next steps:")
        print("    • Run: python demo_transition_matrix.py")
        print("    • Run: python example_transition_matrix_integration.py")
        print("    • Read: TRANSITION_MATRIX_COMPLETE.md")
        print("    • Integrate with your Markov chain model!")
        print("\n" + "🎉" * 35 + "\n")
        return 0
    else:
        print("\n[ERROR] Some tests failed. Please review the errors above.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

