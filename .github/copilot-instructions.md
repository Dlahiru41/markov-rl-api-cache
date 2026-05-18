# Copilot Instructions for markov-rl-api-cache

## Project Overview

This repository implements a **Markov Chain-based Reinforcement Learning framework** for adaptive API caching in microservices. The system uses:
- **Markov chains** to model and predict API request patterns
- **Deep Q-Learning (DQN)** to learn optimal cache policies
- **Redis and in-memory backends** for caching
- **FastAPI** for REST API integration
- **Simulators** for traffic generation and failure injection

## Key Technical Stack
- **Language**: Python 3.8+
- **ML/RL**: PyTorch (≥2.0.0), NumPy, SciPy
- **API Framework**: FastAPI, Uvicorn
- **Cache**: Redis (≥4.5.0)
- **Data**: Pandas, PyArrow, Faker
- **Testing**: pytest (≥7.4.0), pytest-asyncio
- **Visualization**: Matplotlib, Seaborn

## Project Structure

### Source Code (`src/`)
- **`src/markov/`** - Markov chain models (first-order, second-order, context-aware)
  - `transition_matrix.py`, `first_order.py`, `second_order.py`, `context_aware.py`
  - `predictor.py` - Unified prediction interface
  - `evaluation.py` - Model evaluation and visualization
- **`src/rl/`** - Reinforcement learning components
  - `agents/dqn_agent.py` - DQN and Double DQN agents
  - `networks/q_network.py` - Q-network and Dueling Q-network
  - `training/trainer.py` - Training orchestration
  - `actions.py`, `state.py`, `reward.py`, `replay_buffer.py` - Core RL components
- **`src/cache/`** - Cache management
  - `backend.py` - CacheBackend interface, InMemoryBackend
  - `cache_manager.py` - High-level cache operations
  - `redis_backend.py` - Redis integration
  - `prefetch.py` - Prefetch rules
- **`src/integration/`** - System integration
  - `api.py` - FastAPI REST endpoints
  - `controller.py` - Orchestrates Markov + RL + Cache
  - `gym_environment.py` - OpenAI Gym wrapper
- **`src/gateway/`** - API gateway implementation
- **`src/utils/`** - Shared utilities (types, logging, config, exceptions)

### Tests (`tests/`)
- **`tests/unit/`** - Unit tests for individual components
- **`tests/integration/`** - Integration tests for full pipeline
- **`tests/performance/`** - Performance benchmarks

### Configuration
- **`configs/`** - YAML configuration files for training/evaluation/deployment
- **`docker/`** - Docker setup (docker-compose.yml for Redis, Prometheus, Grafana)
- **`simulator/`** - Traffic profiles and failure scenarios

### Documentation (`docs/`)
- **`docs/guides/`** - Setup and usage guides
- **`docs/components/`** - Per-component documentation
- **`docs/reference/`** - Quick reference materials
- **`docs/deployment/`** - Deployment guides
- **`docs/evaluation/`** - Experiment results

## Building and Testing

### Environment Setup
1. **Always install dependencies first**:
   ```bash
   pip install -r requirements.txt
   ```
   For OpenAI Gym support: `pip install -r requirements_gym.txt`
   For integration tests: `pip install -r requirements_integration_tests.txt`

2. **Redis is required for cache backend tests**:
   - Start Redis with Docker: `docker compose -f docker/docker-compose.yml up -d redis`
   - Or install locally: `redis-server`

### Running Tests
- **Run all tests**: `pytest` or `python -m pytest`
- **Run unit tests only**: `pytest tests/unit/`
- **Run integration tests only**: `pytest tests/integration/`
- **Run specific test file**: `pytest tests/unit/test_transition_matrix.py`
- **With verbose output**: `pytest -v`
- **With coverage**: `pytest --cov=src --cov-report=html`

**Important**: Tests are organized in `tests/` as configured in `pytest.ini`. Always validate changes by running relevant test suites.

### Running Demo Scripts
Demo scripts (e.g., `demo_dqn_agent.py`, `demo_trainer.py`) demonstrate component usage:
```bash
python demo_dqn_agent.py
python demo_trainer.py
```

### Running Example Scripts
Integration examples (e.g., `example_dqn_training.py`) show end-to-end workflows:
```bash
python example_dqn_training.py
python example_redis_backend.py
```

## Code Style and Conventions

1. **Type Hints**: Use type hints for function parameters and return values
2. **Imports**: Group imports (standard library, third-party, local)
3. **Docstrings**: Use Google-style docstrings for classes and methods
4. **Error Handling**: Use custom exceptions from `src/utils/exceptions.py`
5. **Logging**: Use the logging utilities from `src/utils/logging.py`
6. **Configuration**: Load configs from YAML files in `configs/`

## Common Patterns

### Working with Markov Chains
```python
from src.markov.first_order import FirstOrderMarkovChain
from src.markov.predictor import MarkovPredictor

# Create and train model
markov = FirstOrderMarkovChain()
markov.train(sequences)

# Make predictions
predictor = MarkovPredictor(markov)
predictions = predictor.predict_next_states(current_state)
```

### Working with DQN Agents
```python
from src.rl.agents.dqn_agent import DQNAgent
from src.rl.training.trainer import Trainer

# Create agent
agent = DQNAgent(state_size, action_size, config)

# Train agent
trainer = Trainer(agent, environment, config)
trainer.train(num_episodes)
```

### Working with Cache
```python
from src.cache.cache_manager import CacheManager
from src.cache.redis_backend import RedisBackend

# Initialize cache
backend = RedisBackend(host='localhost', port=6379)
cache_manager = CacheManager(backend)

# Cache operations
cache_manager.put(key, value, ttl=60)
result = cache_manager.get(key)
```

## Important Notes for Code Changes

1. **Package Structure**: The project uses `src/` layout. Package imports use `src.module.submodule` format.

2. **Dependencies**: Check `requirements.txt` before adding new dependencies. Ensure compatibility with PyTorch 2.0+.

3. **Configuration Files**: Component configurations are stored in `configs/*.yaml`. Always validate YAML syntax.

4. **Integration Tests**: Integration tests may require Redis running. Start with `docker compose -f docker/docker-compose.yml up -d redis`.

5. **Data Files**: Sample data and experiment results are in `data/`, `results/`, and `evaluation/` directories.

6. **Docker Deployment**: Use `docker/docker-compose.yml` for full stack deployment. Scripts in `docker/scripts/` manage services.

7. **Simulator**: The `simulator/` directory contains traffic generators and failure injection tools for testing.

## Validation Steps

Before finalizing code changes:
1. **Run unit tests**: `pytest tests/unit/` - ensure core functionality works
2. **Run integration tests**: `pytest tests/integration/` - validate system integration
3. **Test Redis integration**: Ensure Redis is running for cache-related changes
4. **Run relevant demo scripts**: Verify examples still work
5. **Check type hints**: Ensure new code has proper type annotations
6. **Update documentation**: Add/update docstrings and relevant docs

## Common Issues and Solutions

1. **Module Import Errors**: Ensure you're in the repository root and packages are installed
2. **Redis Connection Errors**: Start Redis with `docker compose -f docker/docker-compose.yml up -d redis`
3. **PyTorch Issues**: Verify PyTorch installation matches requirements (≥2.0.0)
4. **Test Failures**: Check if Redis is required for the failing test
5. **YAML Parse Errors**: Validate YAML syntax in config files

## Trust These Instructions

These instructions have been validated against the current repository state. Only perform additional searches if information is incomplete or incorrect. The structure and commands documented here are the canonical approach for working with this codebase.
