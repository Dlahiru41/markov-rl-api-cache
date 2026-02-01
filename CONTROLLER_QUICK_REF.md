# Integration Controller - Quick Reference

## 🚀 Quick Start

```python
from src.integration.controller import IntegrationController, ControllerConfig

# Create and run
config = ControllerConfig(mode='training', output_dir='results/test')
controller = IntegrationController(config)
controller.setup()
controller.start()
controller.train(num_episodes=100)
controller.stop()
```

## 📋 CLI Commands

```bash
# Train
python scripts/controller.py train --episodes 1000 --output results/run1

# Evaluate
python scripts/controller.py evaluate --model results/run1/best_model.pt --episodes 50

# Serve
python scripts/controller.py serve --model results/run1/best_model.pt --port 8080

# Demo
python scripts/controller.py demo --model results/run1/best_model.pt --interactive

# Status
python scripts/controller.py status --port 8080
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | System status |
| `/metrics` | GET | All metrics |
| `/train/start` | POST | Start training |
| `/train/stop` | POST | Stop training |
| `/train/progress` | GET | Training progress |
| `/evaluate` | POST | Run evaluation |
| `/action` | POST | Get action for state |
| `/api-call` | POST | Process API call |
| `/inject/failure` | POST | Inject failure |
| `/inject/restore` | POST | Restore system |
| `/cache/contents` | GET | View cache |
| `/cache/clear` | DELETE | Clear cache |
| `/config/update` | POST | Update config |

## 🎯 Operating Modes

- **training** - Train RL agents
- **evaluation** - Evaluate trained agents
- **deployment** - Serve predictions via API
- **demo** - Interactive demonstration

## 📊 Key Methods

### Lifecycle
- `setup()` - Initialize components
- `start()` - Begin operation
- `stop()` - Graceful shutdown

### Operations
- `train(num_episodes)` - Train agent
- `evaluate(num_episodes)` - Evaluate agent
- `predict_action(state)` - Get action (deployment)
- `process_api_call(endpoint, context)` - Process API call
- `run_demo(scenario)` - Run demo
- `step_demo()` - Demo step

### Status
- `get_status()` - System status
- `get_metrics()` - All metrics

## 📝 Configuration

```yaml
# configs/default.yaml
mode: training
output_dir: results/default
enable_monitoring: true
enable_api: true

env_config:
  max_steps_per_episode: 200
  
agent_config:
  learning_rate: 0.0001
  batch_size: 64

training_config:
  num_episodes: 1000
  eval_frequency: 50
```

## 🧪 Validation

```bash
# Verify installation
python verify_controller.py

# Run validation tests
python validate_controller.py
```

## 📖 Documentation

- `CONTROLLER_README.md` - Complete guide
- `CONTROLLER_IMPLEMENTATION_COMPLETE.md` - Summary
- `configs/default.yaml` - Config example

## 💡 Common Patterns

### Context Manager
```python
with IntegrationController(config) as ctrl:
    ctrl.train(num_episodes=100)
    ctrl.evaluate(num_episodes=10)
# Auto cleanup
```

### Training Loop
```python
controller.setup()
controller.start()

for i in range(10):
    controller.train(num_episodes=100)
    results = controller.evaluate(num_episodes=10)
    print(f"Iteration {i}: {results['mean_reward']:.2f}")

controller.stop()
```

### API Usage
```bash
# Start server
python scripts/controller.py serve --model model.pt --port 8080

# Call API
curl -X POST http://localhost:8080/api-call \
  -H "Content-Type: application/json" \
  -d '{"endpoint": "/api/products/list", "context": {"user_type": "premium"}}'
```

## 🎉 Features

✅ 4 operating modes (training, evaluation, deployment, demo)  
✅ Complete lifecycle management  
✅ FastAPI control API (15+ endpoints)  
✅ CLI interface (5 commands)  
✅ Prometheus monitoring  
✅ YAML configuration  
✅ Error handling  
✅ Context manager support  
✅ Comprehensive documentation  

## 📁 Files Created

- `src/integration/controller.py` (860+ lines)
- `src/integration/api.py` (400+ lines)
- `scripts/controller.py` (440+ lines)
- `configs/default.yaml`
- `validate_controller.py`
- `CONTROLLER_README.md`
- `CONTROLLER_IMPLEMENTATION_COMPLETE.md`

**Total: ~2,500 lines**

## 🚦 Status

✅ **Implementation Complete**  
✅ **All Files Created**  
✅ **Documentation Complete**  
✅ **Ready to Use**

Start with: `python verify_controller.py`

