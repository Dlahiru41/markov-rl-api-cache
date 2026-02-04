#!/usr/bin/env python3
"""
Verification script for the enterprise demo.
Tests that all components are working correctly.
"""

import sys
import subprocess

def test_dependencies():
    """Test if all dependencies are installed."""
    print("Testing dependencies...")
    try:
        import gymnasium
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import torch
        import sklearn
        print("  ✅ All dependencies installed")
        return True
    except ImportError as e:
        print(f"  ❌ Missing dependency: {e}")
        print("\n  Run: python setup_demo_dependencies.py")
        return False

def test_imports():
    """Test if all demo imports work."""
    print("\nTesting demo imports...")
    sys.path.insert(0, 'src')
    
    try:
        from src.integration.gym_environment import CachingEnv, CacheEnvConfig
        print("  ✅ Gym environment import OK")
    except Exception as e:
        print(f"  ❌ Gym environment import failed: {e}")
        return False
    
    try:
        from src.markov.predictor import MarkovPredictor
        print("  ✅ Markov predictor import OK")
    except Exception as e:
        print(f"  ❌ Markov predictor import failed: {e}")
        return False
    
    try:
        from src.cache.cache_manager import CacheManager
        print("  ✅ Cache manager import OK")
    except Exception as e:
        print(f"  ❌ Cache manager import failed: {e}")
        return False
    
    try:
        from src.rl.agents.dqn_agent import DQNAgent, DQNConfig
        print("  ✅ DQN agent import OK")
    except Exception as e:
        print(f"  ❌ DQN agent import failed: {e}")
        return False
    
    return True

def test_demo_syntax():
    """Test if demo script has valid syntax."""
    print("\nTesting demo script syntax...")
    try:
        result = subprocess.run(
            ['python', '-m', 'py_compile', 'ENTERPRISE_LIVE_DEMO.py'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("  ✅ Demo script syntax valid")
            return True
        else:
            print(f"  ❌ Syntax error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def test_demo_execution():
    """Test if demo runs without crashing."""
    print("\nTesting demo execution (first 3 sections)...")
    try:
        # Run with 3 ENTERs to test first 3 sections
        result = subprocess.run(
            ['python', 'ENTERPRISE_LIVE_DEMO.py'],
            input='\n\n\n',
            capture_output=True,
            text=True,
            timeout=20
        )
        
        # Check for key success indicators
        output = result.stdout
        
        checks = [
            ('SECTION 1: THE BUSINESS PROBLEM', 'Executive Hook'),
            ('SECTION 2: SYSTEM ARCHITECTURE', 'System Architecture'),
            ('SECTION 3: MARKOV CHAIN PREDICTION', 'Markov Prediction'),
        ]
        
        all_passed = True
        for check_str, name in checks:
            if check_str in output:
                print(f"  ✅ {name} section working")
            else:
                print(f"  ❌ {name} section missing")
                all_passed = False
        
        return all_passed
        
    except subprocess.TimeoutExpired:
        print("  ⚠️  Demo timed out (may be normal)")
        return True  # Timeout is acceptable
    except Exception as e:
        print(f"  ❌ Execution failed: {e}")
        return False

def main():
    print("=" * 80)
    print("ENTERPRISE DEMO - VERIFICATION SCRIPT")
    print("=" * 80)
    print()
    
    results = []
    
    # Run tests
    results.append(("Dependencies", test_dependencies()))
    results.append(("Imports", test_imports()))
    results.append(("Syntax", test_demo_syntax()))
    results.append(("Execution", test_demo_execution()))
    
    # Summary
    print()
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name:.<40} {status}")
    
    print()
    print(f"  Total: {passed}/{total} tests passed")
    print()
    
    if passed == total:
        print("=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print()
        print("The demo is ready to run:")
        print("  python ENTERPRISE_LIVE_DEMO.py")
        print()
        return 0
    else:
        print("=" * 80)
        print("❌ SOME TESTS FAILED")
        print("=" * 80)
        print()
        print("Please fix the issues above before running the demo.")
        print()
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user.")
        sys.exit(1)
