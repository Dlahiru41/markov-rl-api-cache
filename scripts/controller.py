"""
Command-line interface for the IntegrationController.

Provides convenient commands for training, evaluation, serving, and demonstration.
"""

import argparse
import sys
import logging
from pathlib import Path
import yaml
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integration.controller import IntegrationController, ControllerConfig
from src.integration.gym_environment import CacheEnvConfig, SimulatorConfig
from src.rl.dqn_agent import DQNConfig
from src.rl.trainer import TrainingConfig

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> ControllerConfig:
    """Load configuration from YAML or JSON file."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load file
    with open(config_file, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)

    # Create config object
    # This is simplified - in production would properly parse nested configs
    return ControllerConfig(**config_dict)


def train_command(args):
    """Execute train command."""
    logger.info("=" * 60)
    logger.info("TRAINING MODE")
    logger.info("=" * 60)

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = ControllerConfig(mode='training')

    # Override config with CLI args
    if args.output:
        config.output_dir = args.output
    if args.model:
        config.agent_model_path = args.model

    config.enable_monitoring = not args.no_monitoring
    config.enable_api = not args.no_api

    if args.port:
        config.api_port = args.port

    # Create and setup controller
    controller = IntegrationController(config)

    if not controller.setup():
        logger.error("Setup failed")
        return 1

    controller.start()

    # Train
    try:
        num_episodes = args.episodes if args.episodes else None
        summary = controller.train(num_episodes=num_episodes)

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Episodes: {summary['num_episodes']}")
        logger.info(f"Mean Reward: {summary['mean_reward']:.2f} ± {summary['std_reward']:.2f}")
        logger.info(f"Final Epsilon: {summary['final_epsilon']:.4f}")
        logger.info(f"Training Time: {summary['training_time_seconds']:.1f}s")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        return 0

    finally:
        controller.stop()


def evaluate_command(args):
    """Execute evaluate command."""
    logger.info("=" * 60)
    logger.info("EVALUATION MODE")
    logger.info("=" * 60)

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = ControllerConfig(mode='evaluation')

    # Override config
    if args.output:
        config.output_dir = args.output
    if args.model:
        config.agent_model_path = args.model
    else:
        logger.error("--model is required for evaluation")
        return 1

    # Create and setup controller
    controller = IntegrationController(config)

    if not controller.setup():
        logger.error("Setup failed")
        return 1

    controller.start()

    # Evaluate
    try:
        results = controller.evaluate(num_episodes=args.episodes)

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"Episodes: {results['num_episodes']}")
        logger.info(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        logger.info(f"Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        logger.info(f"Cache Hit Rate: {results['mean_cache_hit_rate']:.2%}")
        logger.info(f"Cascade Rate: {results['cascade_rate']:.2%} ({results['cascade_count']}/{results['num_episodes']})")
        logger.info("=" * 60)

        # Save results
        results_file = Path(config.output_dir) / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {results_file}")

        return 0

    finally:
        controller.stop()


def serve_command(args):
    """Execute serve command (deployment mode)."""
    logger.info("=" * 60)
    logger.info("DEPLOYMENT MODE")
    logger.info("=" * 60)

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = ControllerConfig(mode='deployment')

    # Override config
    if args.model:
        config.agent_model_path = args.model
    else:
        logger.error("--model is required for deployment")
        return 1

    config.enable_api = True
    config.api_port = args.port
    config.enable_monitoring = not args.no_monitoring

    # Create and setup controller
    controller = IntegrationController(config)

    if not controller.setup():
        logger.error("Setup failed")
        return 1

    controller.start()

    logger.info(f"\nAPI server ready on http://0.0.0.0:{config.api_port}")
    logger.info("Endpoints:")
    logger.info(f"  - Status: http://localhost:{config.api_port}/status")
    logger.info(f"  - Metrics: http://localhost:{config.api_port}/metrics")
    logger.info(f"  - Health: http://localhost:{config.api_port}/health")
    logger.info(f"  - API Docs: http://localhost:{config.api_port}/docs")
    logger.info("\nPress Ctrl+C to stop")

    # Start API server
    try:
        from src.integration.api import serve_api
        serve_api(controller, host="0.0.0.0", port=config.api_port)

    except KeyboardInterrupt:
        logger.info("\nShutting down...")

    finally:
        controller.stop()

    return 0


def demo_command(args):
    """Execute demo command."""
    logger.info("=" * 60)
    logger.info("DEMO MODE")
    logger.info("=" * 60)

    # Load or create config
    if args.config:
        config = load_config(args.config)
    else:
        config = ControllerConfig(mode='demo')

    # Override config
    if args.model:
        config.agent_model_path = args.model
    if args.output:
        config.output_dir = args.output

    # Create and setup controller
    controller = IntegrationController(config)

    if not controller.setup():
        logger.error("Setup failed")
        return 1

    controller.start()

    # Run demo
    try:
        scenario = args.scenario if hasattr(args, 'scenario') else 'normal'

        if args.interactive:
            # Step-by-step mode
            logger.info("\nInteractive step-by-step demo mode")
            logger.info("Press Enter to step, 'q' to quit\n")

            step = 0
            while True:
                user_input = input(f"Step {step} (press Enter or 'q' to quit): ").strip()

                if user_input.lower() == 'q':
                    break

                state = controller.step_demo()

                logger.info(f"\nStep {state['step']}:")
                logger.info(f"  Action: {state['action']}")
                logger.info(f"  Reward: {state['reward']:.2f}")
                logger.info(f"  Cache Hit: {state['cache_hit']}")
                logger.info(f"  Cascade Risk: {state['cascade_risk']:.2%}")

                if state['episode_ended']:
                    logger.info("\n  Episode ended. Starting new episode...\n")

                step += 1
        else:
            # Automatic demo
            results = controller.run_demo(scenario=scenario)

            logger.info("\n" + "=" * 60)
            logger.info("DEMO COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Scenario: {results['scenario']}")
            logger.info(f"Steps: {len(results['steps'])}")
            logger.info(f"Total Reward: {results['total_reward']:.2f}")
            logger.info("=" * 60)

            # Save results
            results_file = Path(config.output_dir) / f"demo_{scenario}_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nDemo results saved to {results_file}")

        return 0

    except KeyboardInterrupt:
        logger.info("\nDemo interrupted")
        return 0

    finally:
        controller.stop()


def status_command(args):
    """Execute status command."""
    # Try to connect to running API server
    try:
        import requests

        url = f"http://localhost:{args.port}/status"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            status = response.json()

            logger.info("=" * 60)
            logger.info("SYSTEM STATUS")
            logger.info("=" * 60)
            logger.info(f"Mode: {status['mode']}")
            logger.info(f"Setup: {status['is_setup']}")
            logger.info(f"Running: {status['is_running']}")
            logger.info(f"Uptime: {status['uptime_seconds']:.1f}s")

            logger.info("\nComponent Health:")
            for component, healthy in status['component_health'].items():
                status_str = "✓" if healthy else "✗"
                logger.info(f"  {status_str} {component}")

            if status.get('training_progress'):
                progress = status['training_progress']
                logger.info("\nTraining Progress:")
                logger.info(f"  Episodes: {progress['episode_count']}")
                logger.info(f"  Steps: {progress['total_steps']}")
                logger.info(f"  Total Reward: {progress['total_reward']:.2f}")

            logger.info("=" * 60)

            return 0
        else:
            logger.error(f"API returned status {response.status_code}")
            return 1

    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to API on port {args.port}")
        logger.info("Is the controller running? Start with: python scripts/controller.py serve")
        return 1
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Integration Controller CLI for Intelligent Caching System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train for 1000 episodes
  python scripts/controller.py train --episodes 1000 --output results/run1
  
  # Evaluate a trained model
  python scripts/controller.py evaluate --model results/run1/best_model.pt --episodes 50
  
  # Start API server in deployment mode
  python scripts/controller.py serve --model results/run1/best_model.pt --port 8080
  
  # Run interactive demo
  python scripts/controller.py demo --model results/run1/best_model.pt --interactive
  
  # Check status of running system
  python scripts/controller.py status --port 8080
        """
    )

    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train command
    train_parser = subparsers.add_parser('train', help='Start training')
    train_parser.add_argument('--config', '-c', help='Path to config file')
    train_parser.add_argument('--model', '-m', help='Path to pre-trained model (resume training)')
    train_parser.add_argument('--output', '-o', help='Output directory')
    train_parser.add_argument('--episodes', '-e', type=int, help='Number of episodes')
    train_parser.add_argument('--port', '-p', type=int, default=8080, help='API port')
    train_parser.add_argument('--no-monitoring', action='store_true', help='Disable monitoring')
    train_parser.add_argument('--no-api', action='store_true', help='Disable API')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    eval_parser.add_argument('--config', '-c', help='Path to config file')
    eval_parser.add_argument('--model', '-m', required=True, help='Path to trained model')
    eval_parser.add_argument('--output', '-o', help='Output directory')
    eval_parser.add_argument('--episodes', '-e', type=int, default=50, help='Number of episodes')

    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start in deployment mode')
    serve_parser.add_argument('--config', '-c', help='Path to config file')
    serve_parser.add_argument('--model', '-m', required=True, help='Path to trained model')
    serve_parser.add_argument('--port', '-p', type=int, default=8080, help='API port')
    serve_parser.add_argument('--no-monitoring', action='store_true', help='Disable monitoring')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument('--config', '-c', help='Path to config file')
    demo_parser.add_argument('--model', '-m', help='Path to trained model')
    demo_parser.add_argument('--output', '-o', help='Output directory')
    demo_parser.add_argument('--scenario', default='normal', help='Demo scenario')
    demo_parser.add_argument('--interactive', '-i', action='store_true', help='Step-by-step mode')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show current status')
    status_parser.add_argument('--port', '-p', type=int, default=8080, help='API port')

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Execute command
    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == 'train':
            return train_command(args)
        elif args.command == 'evaluate':
            return evaluate_command(args)
        elif args.command == 'serve':
            return serve_command(args)
        elif args.command == 'demo':
            return demo_command(args)
        elif args.command == 'status':
            return status_command(args)
        else:
            parser.print_help()
            return 1

    except Exception as e:
        logger.error(f"Command failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

