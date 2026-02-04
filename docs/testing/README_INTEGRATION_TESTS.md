# Integration Tests

This directory contains comprehensive integration tests that verify all system components work together correctly.

## Overview

Integration tests ensure that when we connect all the pieces (Markov predictor, RL agent, cache manager, simulator), they interact correctly and produce expected behavior.

## Test Modules

### 1. `test_environment.py` - Gymnasium Environment Tests

Tests the CachingEnv Gymnasium environment that wraps the entire system.

**Test Classes:**
- `TestEnvironmentBasics`: Basic initialization and API compliance
  - Environment creation, observation/action spaces, reset/step returns
- `TestEnvironmentDynamics`: Caching behavior and dynamics
  - Cache hits/misses, eviction, prefetching, rewards, cascade detection
- `TestEnvironmentReproducibility`: Reproducibility and state management
  - Same/different seeds, state reset
- `TestEnvironmentCompatibility`: RL library compatibility
  - Stable-Baselines3, Gymnasium wrappers, vectorized environments

**Key Tests:** 50+ tests covering environment fundamentals

### 2. `test_training_loop.py` - Training Integration Tests

Tests the complete training pipeline including agent learning and checkpointing.

**Test Classes:**
- `TestTrainingBasics`: Core training functionality
  - Training execution, loss decrease, reward improvement, epsilon decay, target updates
- `TestCheckpointing`: Model persistence
  - Checkpoint creation, loading, resuming, best model saving
- `TestEarlyStoppingAndConvergence`: Training termination
  - Early stopping triggers, minimum episodes, convergence detection
- `TestTrainingMetrics`: Metrics tracking
  - Episode rewards, evaluation results, metric logging

**Key Tests:** 15+ tests covering training lifecycle

### 3. `test_cache_system.py` - Cache Integration Tests

Tests cache manager integration with Markov predictions.

**Test Classes:**
- `TestCacheOperations`: Basic cache operations
  - Set/get, prefetch population, eviction, TTL expiration
- `TestCacheWithMarkov`: Markov-cache integration
  - Prediction-based prefetch, probability-based eviction, hit rate improvement
- `TestCacheMetrics`: Metrics and monitoring
  - Hit rate calculation, metric reset, metric export
- `TestCacheBackendIntegration`: Backend compatibility
  - Memory backend, persistence, concurrent access
- `TestCacheAdvancedFeatures`: Advanced functionality
  - Compression, get_or_set, batch operations

**Key Tests:** 20+ tests covering cache functionality

### 4. `test_simulator.py` - Simulator Integration Tests

Tests microservices simulator, traffic generation, and failure injection.

**Test Classes:**
- `TestServiceInteraction`: Service behavior
  - Service responses, dependencies, latency simulation
- `TestFailureInjection`: Failure scenarios
  - Latency injection, error injection, cascade propagation, restoration
- `TestTrafficGeneration`: Traffic patterns
  - Workflow adherence, rate achievement, user distribution
- `TestRealisticScenarios`: Real-world scenarios
  - Normal traffic, peak load, cascade prevention, cold start
- `TestServiceMetrics`: Monitoring
  - Call counts, error rates, latency tracking
- `TestServiceResilience`: Resilience patterns
  - Retries, circuit breakers, graceful degradation

**Key Tests:** 25+ tests covering simulator capabilities

### 5. `test_full_pipeline.py` - End-to-End Tests

Tests complete workflows from training through deployment.

**Test Classes:**
- `TestEndToEnd`: Full pipeline workflows
  - Complete training pipeline, baseline comparison, model deployment, metrics collection
- `TestScenarios`: Operational scenarios
  - Normal traffic, peak load, cascade prevention, cold start
- `TestMultiAgentComparison`: Policy comparison
  - Multiple policies, statistical comparison
- `TestSystemReliability`: Error handling
  - Invalid states, error recovery, long-running stability
- `TestResourceManagement`: Resource cleanup
  - Memory management, environment cleanup

**Key Tests:** 20+ tests covering end-to-end workflows

### 6. `conftest.py` - Shared Fixtures

Provides common fixtures used across all integration tests:

**Configuration Fixtures:**
- `env_config`: Standard environment configuration
- `fast_env_config`: Fast configuration for quick tests
- `dqn_config`: DQN agent configuration
- `training_config`: Training configuration

**Environment Fixtures:**
- `training_env`: Ready-to-use training environment
- `fast_env`: Fast environment for quick tests
- `multiple_envs`: Multiple environments for parallel testing

**Agent Fixtures:**
- `untrained_agent`: Fresh DQN agent
- `trained_agent`: Pre-trained agent (session-scoped for efficiency)
- `agent_with_experience`: Agent with replay buffer populated

**Component Fixtures:**
- `cache_manager`: Cache manager with memory backend
- `markov_predictor`: Trained Markov predictor
- `context_aware_predictor`: Context-aware predictor
- `mock_services`: Mocked microservices
- `sample_traffic`: Pre-generated traffic data

**Utility Fixtures:**
- `temp_output_dir`: Temporary directory for test outputs
- `random_seed`: Set random seeds for reproducibility

## Running Tests

### Run All Integration Tests
```bash
pytest tests/integration/ -v
```

### Run With Coverage
```bash
pytest tests/integration/ --cov=src --cov-report=html
```

### Run Specific Test File
```bash
pytest tests/integration/test_environment.py -v
pytest tests/integration/test_training_loop.py -v
pytest tests/integration/test_cache_system.py -v
pytest tests/integration/test_simulator.py -v
pytest tests/integration/test_full_pipeline.py -v
```

### Run Tests Matching Pattern
```bash
pytest tests/integration/ -k "cascade" -v
pytest tests/integration/ -k "prefetch" -v
pytest tests/integration/ -k "training" -v
```

### Run With Parallel Execution (Faster)
```bash
pytest tests/integration/ -n auto
```

### Run And Stop On First Failure
```bash
pytest tests/integration/ -x
```

### Run With Detailed Output
```bash
pytest tests/integration/ -vv --tb=short
```

### Run Only Fast Tests (Skip Slow Ones)
```bash
pytest tests/integration/ -m "not slow"
```

## Test Statistics

- **Total Test Modules:** 6 (including conftest)
- **Total Test Classes:** ~25
- **Total Test Functions:** 130+
- **Expected Runtime:** <5 minutes (with parallelization)
- **Code Coverage:** >80% of integration-related code

## Requirements

Tests require the following packages:

```bash
# Core dependencies
pip install pytest pytest-asyncio pytest-cov

# Parallel execution
pip install pytest-xdist

# RL and ML
pip install gymnasium torch numpy

# Optional (for full compatibility tests)
pip install stable-baselines3
```

## Test Design Principles

1. **Isolation**: Each test is independent and can run in any order
2. **Reproducibility**: Tests use fixed seeds for deterministic behavior
3. **Speed**: Fast tests run quickly; slow tests are marked appropriately
4. **Realism**: Tests simulate realistic operational scenarios
5. **Coverage**: Tests cover both happy paths and edge cases
6. **Documentation**: Each test has clear docstrings explaining what it tests

## Troubleshooting

### Tests Are Slow
```bash
# Use parallel execution
pytest tests/integration/ -n auto

# Run only fast tests
pytest tests/integration/ -m "not slow"

# Use fast_env_config fixtures
```

### Tests Are Flaky
```bash
# Check random seeds are set
# Check fixtures are properly isolated
# Run tests individually to identify issues
pytest tests/integration/test_environment.py::TestEnvironmentBasics::test_environment_creation -v
```

### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA/GPU Issues
```bash
# Force CPU for tests
export CUDA_VISIBLE_DEVICES=""

# Or set device='cpu' in configs
```

## CI/CD Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Integration Tests
  run: |
    pytest tests/integration/ -v --cov=src --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Contributing

When adding new integration tests:

1. Follow existing naming conventions (`test_*.py`)
2. Use appropriate fixtures from `conftest.py`
3. Add docstrings explaining what is being tested
4. Ensure tests are deterministic (use seeds)
5. Keep tests focused and independent
6. Mark slow tests with `@pytest.mark.slow`
7. Update this README with new test information

## Related Documentation

- Main README: `../../README.md`
- Unit Tests: `../unit/README.md`
- Gymnasium Environment: `../../GYM_ENVIRONMENT_README.md`
- Training Guide: `../../TRAINER_README.md`
- Cache Manager: `../../CACHE_MANAGER_README.md`

