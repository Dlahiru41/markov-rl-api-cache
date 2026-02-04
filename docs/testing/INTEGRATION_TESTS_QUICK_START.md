# Integration Tests - Quick Start Guide

## 📦 Installation

### Step 1: Install Core Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements_gym.txt
```

### Step 2: Install Test Dependencies
```bash
pip install pytest pytest-cov pytest-xdist pytest-asyncio
```

Or use the integration test requirements file:
```bash
pip install -r requirements_integration_tests.txt
```

### Step 3: Install Package in Development Mode
```bash
pip install -e .
```

## 🚀 Running Tests

### Option 1: Interactive Runner (Recommended)
```bash
python run_integration_tests.py
```

This will show you a menu:
```
1. All integration tests (default)
2. Environment tests only
3. Training loop tests only
4. Cache system tests only
5. Simulator tests only
6. Full pipeline tests only
7. Quick smoke test (fast)
8. All tests with coverage
9. All tests with parallel execution
```

### Option 2: Direct pytest Commands

#### Run All Integration Tests
```bash
pytest tests/integration/ -v
```

#### Run with Coverage Report
```bash
pytest tests/integration/ --cov=src --cov-report=html --cov-report=term
```

View coverage report:
```bash
# Open htmlcov/index.html in your browser
```

#### Run Specific Test File
```bash
# Environment tests
pytest tests/integration/test_environment.py -v

# Training tests
pytest tests/integration/test_training_loop.py -v

# Cache tests
pytest tests/integration/test_cache_system.py -v

# Simulator tests
pytest tests/integration/test_simulator.py -v

# Full pipeline tests
pytest tests/integration/test_full_pipeline.py -v
```

#### Run Specific Test Class
```bash
pytest tests/integration/test_environment.py::TestEnvironmentBasics -v
```

#### Run Specific Test Function
```bash
pytest tests/integration/test_environment.py::TestEnvironmentBasics::test_environment_creation -v
```

#### Run Tests Matching Pattern
```bash
# All tests related to cascades
pytest tests/integration/ -k "cascade" -v

# All tests related to prefetch
pytest tests/integration/ -k "prefetch" -v

# All tests related to training
pytest tests/integration/ -k "training" -v
```

#### Run with Parallel Execution (Faster)
```bash
pytest tests/integration/ -n auto -v
```

#### Run and Stop on First Failure
```bash
pytest tests/integration/ -x
```

#### Run with Detailed Output
```bash
pytest tests/integration/ -vv --tb=long
```

#### Run Only Failed Tests from Last Run
```bash
pytest tests/integration/ --lf
```

## 📊 Expected Output

### Successful Run
```bash
$ pytest tests/integration/ -v

tests/integration/test_environment.py::TestEnvironmentBasics::test_environment_creation PASSED [ 1%]
tests/integration/test_environment.py::TestEnvironmentBasics::test_observation_space_shape PASSED [ 2%]
...
tests/integration/test_full_pipeline.py::TestResourceManagement::test_environment_cleanup PASSED [100%]

========================== 130 passed in 120.45s ==========================
```

### With Coverage
```bash
$ pytest tests/integration/ --cov=src --cov-report=term

---------- coverage: platform win32, python 3.10.x -----------
Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
src/__init__.py                            0      0   100%
src/cache/cache_manager.py              300     45    85%
src/integration/gym_environment.py       450     60    87%
src/rl/agents/dqn_agent.py              200     25    88%
src/rl/training/trainer.py              250     35    86%
...
-----------------------------------------------------------
TOTAL                                  5000    500    90%

========================== 130 passed in 120.45s ==========================
```

## 🎯 What Gets Tested

### Component Integration (130+ tests)

1. **Environment Tests (20+ tests)**
   - Gymnasium API compliance
   - Observation/action spaces
   - Episode dynamics
   - Reproducibility
   - Library compatibility

2. **Training Loop Tests (15+ tests)**
   - Agent training
   - Loss and reward tracking
   - Checkpointing
   - Early stopping
   - Convergence detection

3. **Cache System Tests (20+ tests)**
   - Cache operations
   - Markov prediction integration
   - Prefetching behavior
   - Metrics tracking
   - Backend compatibility

4. **Simulator Tests (25+ tests)**
   - Service interaction
   - Failure injection
   - Traffic generation
   - Realistic scenarios
   - Resilience patterns

5. **Full Pipeline Tests (20+ tests)**
   - End-to-end workflows
   - Baseline comparison
   - Model deployment
   - Statistical validation
   - Resource management

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Solution: Install package in development mode
pip install -e .
```

### Issue: pytest not found
```bash
# Solution: Install pytest
pip install pytest pytest-cov pytest-xdist
```

### Issue: Import errors for src modules
```bash
# Solution: Add to PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%CD%  # Windows
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac

# Or install in development mode
pip install -e .
```

### Issue: Tests running slow
```bash
# Solution: Use parallel execution
pytest tests/integration/ -n auto

# Or run only fast tests
pytest tests/integration/test_environment.py::TestEnvironmentBasics -v
```

### Issue: gymnasium not found
```bash
# Solution: Install gymnasium
pip install gymnasium>=0.29.0
```

### Issue: torch not found
```bash
# Solution: Install PyTorch
pip install torch>=2.0.0
```

### Issue: stable_baselines3 tests skip
```bash
# Solution: Install stable-baselines3 (optional)
pip install stable-baselines3>=2.0.0
```

## 📈 Test Metrics

- **Total Tests:** 130+
- **Test Modules:** 5
- **Test Classes:** 25+
- **Lines of Code:** 2,400+
- **Expected Runtime:** <5 minutes (with parallelization)
- **Expected Coverage:** >80%

## 🎓 Advanced Usage

### Run Tests with Custom Markers
```bash
# Mark tests as slow
@pytest.mark.slow
def test_long_training():
    ...

# Skip slow tests
pytest tests/integration/ -m "not slow"
```

### Generate JUnit XML Report (for CI/CD)
```bash
pytest tests/integration/ --junitxml=test-results.xml
```

### Generate Multiple Coverage Formats
```bash
pytest tests/integration/ --cov=src --cov-report=html --cov-report=xml --cov-report=term
```

### Run with Verbose Logging
```bash
pytest tests/integration/ -v --log-cli-level=INFO
```

### Debug Failing Test
```bash
pytest tests/integration/test_environment.py::TestEnvironmentBasics::test_environment_creation -vv --pdb
```

## 📚 Files Created

```
tests/integration/
├── conftest.py                    # Shared fixtures
├── test_environment.py            # Environment integration tests
├── test_training_loop.py          # Training integration tests
├── test_cache_system.py           # Cache system tests
├── test_simulator.py              # Simulator tests
├── test_full_pipeline.py          # End-to-end tests
└── README_INTEGRATION_TESTS.md    # Detailed documentation

run_integration_tests.py           # Interactive test runner
requirements_integration_tests.txt  # Test dependencies
INTEGRATION_TESTS_COMPLETE.md      # Implementation summary
INTEGRATION_TESTS_QUICK_START.md   # This file
```

## ✅ Validation

To verify everything is set up correctly:

```bash
# 1. Check Python version (should be 3.8+)
python --version

# 2. Check pytest is installed
pytest --version

# 3. Verify package can be imported
python -c "from src.integration.gym_environment import CachingEnv; print('✅ Import successful')"

# 4. Run quick smoke test
pytest tests/integration/test_environment.py::TestEnvironmentBasics::test_environment_creation -v

# 5. If all above pass, run full suite
pytest tests/integration/ -v
```

## 🎉 You're Ready!

If all validation steps pass, you're ready to run the comprehensive integration tests:

```bash
python run_integration_tests.py
```

Or directly with pytest:

```bash
pytest tests/integration/ -v --cov=src --cov-report=html
```

Good luck with your testing! 🚀

