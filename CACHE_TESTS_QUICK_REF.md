# Cache Tests - Quick Reference

## 🚀 Quick Start

### Prerequisites
```bash
# Install test dependencies
pip install pytest pytest-cov

# Install Redis (optional, for integration tests)
docker run -d -p 6379:6379 redis
```

### Run All Tests
```bash
# Using test runner (recommended)
python run_cache_tests.py

# Using pytest
pytest tests/ -v
```

## 📁 Test Organization

```
tests/
├── unit/                           # Unit tests (no external dependencies)
│   ├── test_cache_backend.py      # InMemory backend (52 tests)
│   └── test_redis_backend.py      # Redis backend mocked (37 tests)
├── integration/                    # Integration tests (requires Redis)
│   ├── test_redis_integration.py  # Real Redis tests (37 tests)
│   └── test_cache_backend_comparison.py  # Backend comparison (46 test runs)
└── performance/                    # Performance benchmarks
    └── test_cache_performance.py  # Performance tests (25 tests)
```

## 🎯 Common Commands

### By Test Type
```bash
# Unit tests (fast, no Redis needed)
pytest tests/unit/ -v

# Integration tests (requires Redis)
pytest tests/integration/ -v

# Performance tests (with output)
pytest tests/performance/ -v -s
```

### By Component
```bash
# InMemory backend tests
pytest tests/unit/test_cache_backend.py -v

# Redis backend tests (mocked)
pytest tests/unit/test_redis_backend.py -v

# Redis integration tests (real Redis)
pytest tests/integration/test_redis_integration.py -v

# Backend comparison
pytest tests/integration/test_cache_backend_comparison.py -v
```

### Specific Test Classes
```bash
# Test TTL functionality
pytest tests/integration/test_redis_integration.py::TestRedisTTL -v

# Test batch operations
pytest tests/integration/test_redis_integration.py::TestRedisBatchOperations -v

# Test thread safety
pytest tests/integration/test_redis_integration.py::TestRedisThreadSafety -v

# Test performance
pytest tests/performance/test_cache_performance.py::TestSingleOperationPerformance -v
```

### Specific Tests
```bash
# Run single test
pytest tests/unit/test_cache_backend.py::TestCacheEntry::test_create_entry_basic -v
```

## 🔍 Test Filtering

```bash
# Run tests matching pattern
pytest tests/ -k "ttl" -v

# Run tests NOT matching pattern
pytest tests/ -k "not redis" -v

# Run only failed tests from last run
pytest --lf -v
```

## 📊 Coverage Reports

```bash
# Run with coverage
pytest tests/ --cov=src/cache --cov-report=term

# Generate HTML coverage report
pytest tests/ --cov=src/cache --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser
```

## 🐛 Debugging

```bash
# Show print statements
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Enter debugger on failure
pytest tests/ --pdb

# Verbose output
pytest tests/ -vv
```

## ⚡ Performance Tests

```bash
# Run all performance tests
pytest tests/performance/ -v -s

# Run specific performance category
pytest tests/performance/test_cache_performance.py::TestThroughputPerformance -v -s

# Quick performance check
pytest tests/performance/ -k "throughput" -v -s
```

## 🔧 Test Configuration

### pytest.ini
```ini
[pytest]
minversion = 7.0
addopts = -ra -q
testpaths = tests
python_files = test_*.py *_test.py
```

### Skip Tests Without Redis
```bash
# Automatically skips if Redis unavailable
pytest tests/integration/test_redis_integration.py -v
```

## 📈 Test Counts

| Category | File | Tests | Notes |
|----------|------|-------|-------|
| Unit | test_cache_backend.py | 52 | InMemory + abstractions |
| Unit | test_redis_backend.py | 37 | Redis mocked |
| Integration | test_redis_integration.py | 37 | Requires Redis |
| Integration | test_cache_backend_comparison.py | 23 × 2 | Both backends |
| Performance | test_cache_performance.py | 25 | Benchmarks |
| **TOTAL** | | **197** | |

## 🎯 Test Selection Examples

### Development Workflow
```bash
# 1. Quick unit tests during development
pytest tests/unit/ -v

# 2. Full validation before commit
pytest tests/ -v

# 3. Performance check before release
pytest tests/performance/ -v -s
```

### CI/CD Pipeline
```bash
# Stage 1: Unit tests (fast)
pytest tests/unit/ -v --tb=short

# Stage 2: Integration tests (with Redis)
pytest tests/integration/ -v --tb=short

# Stage 3: Performance tests (optional)
pytest tests/performance/ -v -s --tb=short
```

## 🚦 Test Output Examples

### Success
```
tests/unit/test_cache_backend.py::TestCacheEntry::test_create_entry_basic PASSED [1%]
tests/unit/test_cache_backend.py::TestCacheEntry::test_create_entry_with_ttl PASSED [2%]
...
================================ 52 passed in 2.34s ================================
```

### With Coverage
```
----------- coverage: platform win32, python 3.x -----------
Name                          Stmts   Miss  Cover
-------------------------------------------------
src/cache/backend.py            200      0   100%
src/cache/redis_backend.py      300      0   100%
-------------------------------------------------
TOTAL                           500      0   100%
```

### Performance Output
```
InMemory SET performance:
  Mean: 0.0234 ms
  Median: 0.0198 ms
  Min: 0.0145 ms
  Max: 0.1234 ms

Redis SET performance:
  Mean: 1.2345 ms
  Median: 1.1234 ms
  Min: 0.8912 ms
  Max: 5.6789 ms
```

## 📝 Quick Validation

```bash
# Without pytest (basic validation)
python validate_cache_tests.py
```

## 🎉 Full Test Run

```bash
# Comprehensive test runner
python run_cache_tests.py
```

Output:
```
======================================================================
Running: Unit Tests - Cache Backend (InMemory)
======================================================================
✅ PASSED - Cache Backend Unit Tests

======================================================================
Running: Unit Tests - Redis Backend
======================================================================
✅ PASSED - Redis Backend Unit Tests

======================================================================
Running: Integration Tests - Redis Backend (requires Redis server)
======================================================================
✅ PASSED - Redis Integration Tests

======================================================================
Running: Integration Tests - Backend Comparison
======================================================================
✅ PASSED - Backend Comparison Tests

======================================================================
Running: Performance Tests
======================================================================
✅ PASSED - Performance Tests

======================================================================
TEST SUMMARY
======================================================================
✅ PASSED - Cache Backend Unit Tests
✅ PASSED - Redis Backend Unit Tests
✅ PASSED - Redis Integration Tests
✅ PASSED - Backend Comparison Tests
✅ PASSED - Performance Tests

======================================================================
🎉 ALL TESTS PASSED!
======================================================================
```

## 🔗 Related Documentation

- **CACHE_TESTS_COMPLETE.md** - Comprehensive test documentation
- **CACHE_TEST_SUITE_SUMMARY.md** - Implementation summary
- **CACHE_BACKEND_README.md** - Implementation guide
- **REDIS_BACKEND_README.md** - Redis backend guide

## 💡 Tips

1. **Run unit tests frequently** - They're fast and catch most issues
2. **Run integration tests before commit** - Validates real behavior
3. **Run performance tests occasionally** - Ensures no regressions
4. **Use `-v` flag** - See which tests are running
5. **Use `-s` flag** - See print output for debugging
6. **Use `--lf` flag** - Rerun only failed tests
7. **Use `-k` flag** - Filter tests by name pattern

## ⚠️ Troubleshooting

### Redis Connection Failed
```
Solution: Start Redis server
docker run -d -p 6379:6379 redis
```

### Import Errors
```
Solution: Install dependencies
pip install -r requirements.txt
```

### Performance Test Failures
```
Solution: System may be under load. Run again or adjust thresholds.
```

### pytest Not Found
```
Solution: Install pytest
pip install pytest
```

---

**Quick Commands Summary:**

```bash
# Most common commands
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/performance/ -v -s          # Performance tests
python run_cache_tests.py                # All tests
python validate_cache_tests.py           # Quick validation
```

🎯 **Default**: `pytest tests/ -v` - Runs all tests with verbose output

