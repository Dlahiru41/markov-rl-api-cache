# Markov Chain Test Suite Documentation

## Overview

Comprehensive pytest test suite for all Markov chain components with >85% code coverage.

## Test Structure

```
tests/
├── __init__.py
└── unit/
    ├── __init__.py
    └── test_markov.py  (Main test file - 800+ lines)
```

## Test Classes

### 1. TestTransitionMatrix (9 tests)
Tests for the core TransitionMatrix data structure:
- ✅ `test_increment_single` - Adding one transition
- ✅ `test_increment_multiple` - Accumulating counts
- ✅ `test_probability_calculation` - Correct probabilities
- ✅ `test_smoothing` - Laplace smoothing
- ✅ `test_top_k` - Top-k selection
- ✅ `test_top_k_handles_ties` - Deterministic tie-breaking
- ✅ `test_serialization_roundtrip` - Save/load
- ✅ `test_unknown_state` - Graceful error handling
- ✅ `test_merge` - Matrix merging

### 2. TestFirstOrderMarkovChain (8 tests)
Tests for first-order Markov chains:
- ✅ `test_fit_creates_transitions` - Training works
- ✅ `test_predict_returns_sorted` - Predictions ordered
- ✅ `test_predict_unknown_state` - Unknown state handling
- ✅ `test_partial_fit_accumulates` - Incremental learning
- ✅ `test_evaluate_metrics` - Metrics computation
- ✅ `test_generate_sequence` - Sequence generation
- ✅ `test_score_sequence` - Sequence scoring
- ✅ `test_save_load` - Model persistence

### 3. TestSecondOrderMarkovChain (4 tests)
Tests for second-order Markov chains:
- ✅ `test_uses_both_states` - Uses previous + current
- ✅ `test_fallback_to_first_order` - Fallback mechanism
- ✅ `test_fallback_disabled` - Fallback off behavior
- ✅ `test_compare_with_first_order` - Comparison method

### 4. TestContextAwareMarkovChain (4 tests)
Tests for context-aware chains:
- ✅ `test_different_contexts_different_predictions` - Context affects output
- ✅ `test_context_discretization` - Hour to time_of_day
- ✅ `test_global_fallback` - Unknown context handling
- ✅ `test_context_statistics` - Statistics method

### 5. TestMarkovPredictor (7 tests)
Tests for the unified predictor interface:
- ✅ `test_unified_interface_order1` - First-order
- ✅ `test_unified_interface_order2` - Second-order
- ✅ `test_unified_interface_context_aware` - Context-aware
- ✅ `test_history_management` - History tracking
- ✅ `test_state_vector_shape` - Fixed-size vectors
- ✅ `test_state_vector_values` - Vector contents
- ✅ `test_metrics_tracking` - Metrics recording

### 6. TestEdgeCases (5 tests)
Edge case and error handling:
- ✅ `test_empty_sequences` - Empty input
- ✅ `test_single_element_sequences` - Length-1 sequences
- ✅ `test_very_long_sequences` - Performance (marked slow)
- ✅ `test_unicode_endpoints` - Non-ASCII names
- ✅ `test_special_characters_in_endpoints` - Special chars

### 7. TestParametrized (3 test groups)
Parametrized tests for multiple configurations:
- ✅ `test_different_orders` - Order 1 and 2
- ✅ `test_different_smoothing_values` - Various smoothing
- ✅ `test_different_k_values` - Various k values

### 8. TestIntegration (3 tests)
End-to-end integration tests:
- ✅ `test_end_to_end_workflow` - Complete workflow
- ✅ `test_model_persistence_workflow` - Save/load workflow
- ✅ `test_context_aware_end_to_end` - Context-aware workflow

### 9. TestPerformance (2 tests - marked slow)
Performance tests:
- ✅ `test_large_vocabulary_performance` - 1000 unique APIs
- ✅ `test_many_sequences_performance` - 1000 sequences

### 10. TestFactoryFunction (2 tests)
Factory function tests:
- ✅ `test_create_predictor_from_config` - Config-based creation
- ✅ `test_create_predictor_context_aware_config` - Context-aware config

**Total: 47+ tests**

## Running Tests

### Run All Tests
```bash
pytest tests/unit/test_markov.py -v
```

### Run Only Fast Tests (Skip Slow)
```bash
pytest tests/unit/test_markov.py -v -m "not slow"
```

### Run with Coverage Report
```bash
pytest tests/unit/test_markov.py --cov=src/markov --cov-report=html --cov-report=term
```

### Run Specific Test Class
```bash
pytest tests/unit/test_markov.py::TestTransitionMatrix -v
```

### Run Specific Test
```bash
pytest tests/unit/test_markov.py::TestTransitionMatrix::test_increment_single -v
```

### Using Test Runner Script
```bash
# All tests
python run_tests.py

# Fast tests only
python run_tests.py --fast

# With coverage
python run_tests.py --coverage
```

## Test Fixtures

### `simple_sequences`
Small dataset for quick tests:
```python
[
    ['login', 'profile', 'orders'],
    ['login', 'browse', 'product'],
    ['browse', 'product', 'cart'],
    ['login', 'profile', 'settings']
]
```

### `large_sequences`
100 sequences with realistic patterns:
- 40% login flow
- 30% browse flow
- 20% direct checkout
- 10% exploration

### `sequences_with_context`
Sequences paired with context dictionaries:
```python
sequences = [['login', 'premium_features', 'advanced'], ...]
contexts = [{'user_type': 'premium', 'hour': 10}, ...]
```

### `temp_dir`
Temporary directory for file I/O tests.

## Pytest Markers

### `@pytest.mark.slow`
Tests that take >1 second:
- Very long sequences (1000 elements)
- Large vocabulary (1000 unique APIs)
- Performance benchmarks

Skip with: `pytest -m "not slow"`

## Coverage Goals

Target: **>85% code coverage**

Check coverage:
```bash
pytest tests/unit/test_markov.py --cov=src/markov --cov-report=term-missing
```

HTML report:
```bash
pytest tests/unit/test_markov.py --cov=src/markov --cov-report=html
# Open htmlcov/index.html
```

## Test Organization

### Unit Tests
- Isolated component testing
- Fast execution (<0.1s per test)
- No external dependencies

### Integration Tests
- Multi-component workflows
- End-to-end scenarios
- File I/O operations

### Performance Tests
- Marked as `@pytest.mark.slow`
- Benchmark large inputs
- Verify reasonable speed

### Parametrized Tests
- Test multiple configurations
- DRY (Don't Repeat Yourself)
- Comprehensive coverage

## Common Test Patterns

### Basic Test Structure
```python
def test_feature_name(self, fixture):
    """What the test validates."""
    # Arrange
    component = Component(param=value)
    
    # Act
    result = component.method()
    
    # Assert
    assert result == expected_value
```

### Testing with Fixtures
```python
def test_with_data(self, simple_sequences):
    """Test using fixture data."""
    mc = FirstOrderMarkovChain()
    mc.fit(simple_sequences)
    assert mc.is_fitted
```

### Testing Exceptions
```python
def test_invalid_input(self):
    """Test error handling."""
    with pytest.raises(ValueError):
        component.method(invalid_param)
```

### Parametrized Tests
```python
@pytest.mark.parametrize("order", [1, 2])
def test_both_orders(self, order):
    """Test for multiple values."""
    predictor = MarkovPredictor(order=order)
    assert predictor.order == order
```

## What Each Test Validates

### Correctness Tests
- Algorithms produce correct outputs
- Probabilities sum to 1
- Predictions are sorted
- Metrics are accurate

### Robustness Tests
- Empty inputs don't crash
- Unknown states handled gracefully
- Edge cases work correctly
- Special characters supported

### Performance Tests
- Large inputs complete quickly
- Memory usage is reasonable
- No O(n²) bottlenecks

### Persistence Tests
- Save/load roundtrip works
- Loaded models match originals
- File formats are correct

### Integration Tests
- Components work together
- Workflows complete successfully
- State is managed correctly

## Debugging Failed Tests

### View Full Traceback
```bash
pytest tests/unit/test_markov.py -v --tb=long
```

### Stop at First Failure
```bash
pytest tests/unit/test_markov.py -x
```

### Run Last Failed Tests
```bash
pytest tests/unit/test_markov.py --lf
```

### Print Statements
```bash
pytest tests/unit/test_markov.py -s
```

### Verbose Output
```bash
pytest tests/unit/test_markov.py -vv
```

## Continuous Integration

For CI/CD pipelines:
```bash
# Run tests and generate coverage
pytest tests/unit/test_markov.py \
    --cov=src/markov \
    --cov-report=xml \
    --cov-report=term \
    --junitxml=test-results.xml

# Exit code 0 = all passed
# Exit code 1 = failures
```

## Test Development Tips

1. **Write tests first** (TDD) when fixing bugs
2. **Use fixtures** for common setup
3. **Parametrize** instead of copy-paste
4. **Mark slow tests** with `@pytest.mark.slow`
5. **Test edge cases**: empty, single, very large
6. **Verify error handling** with `pytest.raises`
7. **Check coverage** regularly
8. **Keep tests fast**: <0.1s per test ideal

## Expected Test Results

When all tests pass, you should see:
```
====== test session starts ======
collected 47+ items

tests/unit/test_markov.py::TestTransitionMatrix::test_increment_single PASSED
tests/unit/test_markov.py::TestTransitionMatrix::test_increment_multiple PASSED
...
tests/unit/test_markov.py::TestFactoryFunction::test_create_predictor_context_aware_config PASSED

====== 47 passed in X.XXs ======
```

With coverage:
```
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
src/markov/__init__.py                8      0   100%
src/markov/transition_matrix.py     120      5    96%
src/markov/first_order.py           150      8    95%
src/markov/second_order.py          180     12    93%
src/markov/context_aware.py         220     18    92%
src/markov/predictor.py             180     10    94%
---------------------------------------------------------------
TOTAL                               858     53    94%
```

## Troubleshooting

### Tests Not Found
- Check directory structure: `tests/unit/test_markov.py`
- Ensure `__init__.py` files exist
- Verify imports work

### Import Errors
- Install requirements: `pip install -r requirements.txt`
- Check `PYTHONPATH` includes project root
- Use `python -m pytest` instead of `pytest`

### Slow Tests
- Run without slow tests: `pytest -m "not slow"`
- Reduce dataset sizes in fixtures
- Use `pytest-xdist` for parallel execution

### Coverage Not Working
- Install: `pip install pytest-cov`
- Specify source: `--cov=src/markov`
- Check paths are correct

---

**Status:** ✅ Complete Test Suite  
**Tests:** 47+ tests covering all components  
**Coverage Target:** >85%  
**Run Time:** ~5 seconds (fast tests), ~15 seconds (all tests)

