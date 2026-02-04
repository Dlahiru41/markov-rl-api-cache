# Integration Controller - Complete Documentation

## Overview

The **IntegrationController** is the main orchestrator for the intelligent caching system. It manages the lifecycle of all components (Markov predictor, RL agent, cache manager, Gymnasium environment) and provides a unified interface for training, evaluation, and deployment.

## Architecture

```
IntegrationController
├── MarkovPredictor (API call prediction)
├── CacheManager (cache operations)
├── CachingEnv (Gymnasium environment)
├── DQNAgent (RL agent)
├── Trainer (training orchestration)
├── FastAPI (optional control API)
└── Prometheus (optional monitoring)
```

## Installation

### Core Dependencies
```bash
pip install gymnasium numpy torch pyyaml
```

### Optional Dependencies
```bash
# For API
pip install fastapi uvicorn requests

# For monitoring
pip install prometheus_client

# For advanced logging
pip install wandb tensorboard
```

## Quick Start

### 1. Training

```python
from src.integration.controller import IntegrationController, ControllerConfig

# Create controller
config = ControllerConfig(
    mode='training',
    output_dir='results/my_run',
    enable_monitoring=True,
    enable_api=True
)

controller = IntegrationController(config)

# Setup and train
controller.setup()
controller.start()
summary = controller.train(num_episodes=100)

print(f"Training complete: {summary}")

controller.stop()
```

### 2. Using CLI

```bash
# Train
python scripts/controller.py train --episodes 1000 --output results/run1

# Evaluate
python scripts/controller.py evaluate --model results/run1/best_model.pt --episodes 50

# Serve (deployment)
python scripts/controller.py serve --model results/run1/best_model.pt --port 8080

# Demo
python scripts/controller.py demo --model results/run1/best_model.pt --interactive
```

## Configuration

### ControllerConfig

```python
@dataclass
class ControllerConfig:
    # Operating mode
    mode: str = "training"  # training, evaluation, deployment, demo
    
    # Component configurations
    env_config: Optional[CacheEnvConfig] = None
    agent_config: Optional[DQNConfig] = None
    training_config: Optional[TrainingConfig] = None
    
    # Model paths
    markov_model_path: Optional[str] = None
    agent_model_path: Optional[str] = None
    
    # Output
    output_dir: str = "results/default"
    
    # Features
    enable_monitoring: bool = False
    enable_api: bool = False
    log_level: str = "INFO"
    
    # API settings
    api_port: int = 8080
    api_host: str = "0.0.0.0"
```

### YAML Configuration

See `configs/default.yaml` for a complete example:

```yaml
mode: training
output_dir: results/default_run
enable_monitoring: true
enable_api: true

env_config:
  max_steps_per_episode: 200
  simulator_config:
    num_apis: 20
    session_length_range: [10, 100]

agent_config:
  hidden_dims: [128, 128]
  learning_rate: 0.0001
  batch_size: 64

training_config:
  num_episodes: 1000
  eval_frequency: 50
```

Load with:
```python
config = ControllerConfig.load('configs/default.yaml')
```

## Operating Modes

### Training Mode

Train an RL agent to make intelligent caching decisions.

```python
config = ControllerConfig(mode='training')
controller = IntegrationController(config)
controller.setup()
controller.start()

# Train for 1000 episodes
summary = controller.train(num_episodes=1000)

# Periodically evaluate
eval_results = controller.evaluate(num_episodes=10)
```

**CLI:**
```bash
python scripts/controller.py train \
    --config configs/default.yaml \
    --episodes 1000 \
    --output results/run1
```

### Evaluation Mode

Evaluate a trained agent's performance.

```python
config = ControllerConfig(
    mode='evaluation',
    agent_model_path='results/run1/best_model.pt'
)
controller = IntegrationController(config)
controller.setup()

results = controller.evaluate(num_episodes=50)
print(f"Mean reward: {results['mean_reward']:.2f}")
print(f"Cache hit rate: {results['mean_cache_hit_rate']:.2%}")
```

**CLI:**
```bash
python scripts/controller.py evaluate \
    --model results/run1/best_model.pt \
    --episodes 50 \
    --output results/eval
```

### Deployment Mode

Serve predictions via REST API for production use.

```python
config = ControllerConfig(
    mode='deployment',
    agent_model_path='results/run1/best_model.pt',
    enable_api=True,
    api_port=8080
)
controller = IntegrationController(config)
controller.setup()
controller.start()

# API is now serving on http://0.0.0.0:8080
```

**CLI:**
```bash
python scripts/controller.py serve \
    --model results/run1/best_model.pt \
    --port 8080
```

### Demo Mode

Interactive demonstration with visualization.

```python
config = ControllerConfig(mode='demo')
controller = IntegrationController(config)
controller.setup()

# Run automatic demo
results = controller.run_demo(scenario='normal')

# Or step-by-step
for _ in range(100):
    state = controller.step_demo()
    print(f"Step {state['step']}: {state['action']} -> reward {state['reward']:.2f}")
```

**CLI:**
```bash
# Automatic demo
python scripts/controller.py demo --model results/run1/best_model.pt

# Interactive step-by-step
python scripts/controller.py demo --model results/run1/best_model.pt --interactive
```

## API Endpoints

When `enable_api=True`, the following REST endpoints are available:

### Status & Health

- `GET /` - Service info
- `GET /health` - Health check
- `GET /status` - Detailed system status
- `GET /metrics` - Comprehensive metrics

### Training

- `POST /train/start` - Start training
  ```json
  {"num_episodes": 100}
  ```

- `POST /train/stop` - Stop training

- `GET /train/progress` - Training progress

### Evaluation

- `POST /evaluate` - Run evaluation
  ```json
  {"num_episodes": 10}
  ```

### Deployment

- `POST /action` - Get action for state
  ```json
  {"state": [0.1, 0.2, ...]}
  ```

- `POST /api-call` - Process API call
  ```json
  {
    "endpoint": "/api/products/list",
    "context": {"user_type": "premium", "hour": 14}
  }
  ```

### Testing

- `POST /inject/failure` - Inject failure scenario
  ```json
  {"failure_type": "cascade", "severity": 0.8}
  ```

- `POST /inject/restore` - Restore from failure

### Cache Management

- `GET /cache/contents` - View cache
- `DELETE /cache/clear` - Clear cache

### Configuration

- `POST /config/update` - Update configuration
  ```json
  {"config": {"log_level": "DEBUG"}}
  ```

## Monitoring

When `enable_monitoring=True`, Prometheus metrics are exposed:

- `markov_prediction_accuracy` - Markov predictor accuracy
- `cache_hit_rate` - Cache hit rate
- `cache_utilization` - Cache utilization
- `rl_episode_reward` - RL episode reward
- `rl_epsilon` - Exploration rate
- `system_latency_p99` - P99 latency
- `cascade_risk_score` - Cascade failure risk
- `total_episodes` - Total episodes completed

Access metrics at `http://localhost:8080/metrics` (JSON format).

For Prometheus scraping, configure:
```yaml
scrape_configs:
  - job_name: 'caching_system'
    static_configs:
      - targets: ['localhost:8080']
```

## Methods

### Lifecycle Management

#### `setup() -> bool`
Initialize all components in correct order. Returns True if successful.

```python
if controller.setup():
    print("Ready to go!")
```

#### `start()`
Begin active operation based on mode.

```python
controller.start()
```

#### `stop()`
Gracefully shutdown all components, save state.

```python
controller.stop()  # Always call this!
```

### Training & Evaluation

#### `train(num_episodes=None) -> Dict`
Run training for specified episodes.

```python
summary = controller.train(num_episodes=100)
# Returns: {num_episodes, mean_reward, std_reward, final_epsilon, ...}
```

#### `evaluate(num_episodes=10) -> Dict`
Run evaluation with greedy policy.

```python
results = controller.evaluate(num_episodes=50)
# Returns: {mean_reward, cache_hit_rate, cascade_rate, ...}
```

### Deployment

#### `predict_action(state) -> int`
Get action for state (deployment mode).

```python
action = controller.predict_action(state_vector)
```

#### `process_api_call(endpoint, context=None) -> Dict`
Complete API call processing pipeline.

```python
result = controller.process_api_call(
    endpoint='/api/products/list',
    context={'user_type': 'premium'}
)
# Returns: {predictions, action_taken, timestamp}
```

### Demo

#### `run_demo(scenario='normal') -> Dict`
Run automatic demonstration.

```python
results = controller.run_demo(scenario='cascade')
```

#### `step_demo() -> Dict`
Execute one demo step (for interactive visualization).

```python
state = controller.step_demo()
print(f"Action: {state['action']}, Reward: {state['reward']}")
```

### Status & Metrics

#### `get_status() -> Dict`
Get current system status.

```python
status = controller.get_status()
# Returns: {is_setup, is_running, mode, component_health, training_progress, ...}
```

#### `get_metrics() -> Dict`
Get comprehensive metrics from all components.

```python
metrics = controller.get_metrics()
# Returns: {markov, cache, agent, training, environment}
```

## CLI Reference

### Global Options
- `--verbose, -v` - Verbose logging

### Commands

#### `train`
Start training.

**Options:**
- `--config, -c PATH` - Config file
- `--model, -m PATH` - Pre-trained model (resume)
- `--output, -o DIR` - Output directory
- `--episodes, -e NUM` - Number of episodes
- `--port, -p PORT` - API port (default: 8080)
- `--no-monitoring` - Disable monitoring
- `--no-api` - Disable API

**Example:**
```bash
python scripts/controller.py train \
    --config configs/default.yaml \
    --episodes 1000 \
    --output results/run1 \
    --port 8080
```

#### `evaluate`
Run evaluation.

**Options:**
- `--config, -c PATH` - Config file
- `--model, -m PATH` - Trained model (required)
- `--output, -o DIR` - Output directory
- `--episodes, -e NUM` - Number of episodes (default: 50)

**Example:**
```bash
python scripts/controller.py evaluate \
    --model results/run1/best_model.pt \
    --episodes 100 \
    --output results/eval
```

#### `serve`
Start deployment mode.

**Options:**
- `--config, -c PATH` - Config file
- `--model, -m PATH` - Trained model (required)
- `--port, -p PORT` - API port (default: 8080)
- `--no-monitoring` - Disable monitoring

**Example:**
```bash
python scripts/controller.py serve \
    --model results/run1/best_model.pt \
    --port 8080
```

#### `demo`
Run demonstration.

**Options:**
- `--config, -c PATH` - Config file
- `--model, -m PATH` - Trained model
- `--output, -o DIR` - Output directory
- `--scenario STR` - Scenario (default: normal)
- `--interactive, -i` - Step-by-step mode

**Example:**
```bash
# Automatic demo
python scripts/controller.py demo --model results/run1/best_model.pt

# Interactive
python scripts/controller.py demo --model results/run1/best_model.pt --interactive
```

#### `status`
Show system status.

**Options:**
- `--port, -p PORT` - API port (default: 8080)

**Example:**
```bash
python scripts/controller.py status --port 8080
```

## Context Manager

Use as context manager for automatic cleanup:

```python
config = ControllerConfig(mode='training')

with IntegrationController(config) as controller:
    summary = controller.train(num_episodes=100)
    results = controller.evaluate(num_episodes=10)

# Automatically stopped and cleaned up
```

## Error Handling

The controller handles errors gracefully:

```python
try:
    controller.setup()
    controller.start()
    controller.train(num_episodes=1000)
except KeyboardInterrupt:
    print("Training interrupted by user")
except Exception as e:
    print(f"Error: {e}")
finally:
    controller.stop()  # Always cleanup
```

## Best Practices

### 1. Always Setup First
```python
controller = IntegrationController(config)
controller.setup()  # Always call before start()
controller.start()
```

### 2. Always Stop/Cleanup
```python
try:
    controller.train(...)
finally:
    controller.stop()  # Ensures proper cleanup
```

### 3. Check Status
```python
if not controller.setup():
    print("Setup failed!")
    return

status = controller.get_status()
if not all(status['component_health'].values()):
    print("Some components unhealthy!")
```

### 4. Use Config Files
```python
# Better than hardcoding
config = ControllerConfig.load('configs/production.yaml')
```

### 5. Monitor Training
```python
for episode in range(0, 1000, 100):
    controller.train(num_episodes=100)
    eval_results = controller.evaluate(num_episodes=10)
    print(f"Episode {episode}: {eval_results['mean_reward']:.2f}")
```

## Examples

See complete examples in:
- `validate_controller.py` - Validation script
- `scripts/controller.py` - CLI implementation
- `configs/default.yaml` - Example configuration

## Troubleshooting

### "Setup failed"
- Check log output for specific component failure
- Verify all dependencies installed
- Check file paths for models

### "Cannot train in X mode"
- Ensure mode is set to 'training'
- Call `setup()` before `train()`

### "Agent model path specified but not found"
- Verify file exists
- Use absolute path or path relative to working directory

### API won't start
- Install FastAPI: `pip install fastapi uvicorn`
- Check port not already in use
- Verify `enable_api=True`

### Monitoring not working
- Install prometheus_client: `pip install prometheus_client`
- Verify `enable_monitoring=True`

## Next Steps

1. **Run validation:** `python validate_controller.py`
2. **Try CLI:** `python scripts/controller.py train --episodes 10`
3. **Read examples:** Check `validate_controller.py`
4. **Configure:** Edit `configs/default.yaml`
5. **Train:** Run full training session
6. **Deploy:** Start API server with trained model

## License

[Your License]

## Support

For issues or questions, see the main project README or open an issue.

