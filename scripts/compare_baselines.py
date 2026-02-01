"""
Command-line tool for comparing baseline caching policies.

This script provides a comprehensive CLI for running baseline comparisons,
generating reports, and visualizing results.

Usage:
    python scripts/compare_baselines.py --env-config configs/default.yaml --episodes 100 --output results/comparison

    python scripts/compare_baselines.py --baselines lru,lfu,static_markov,random --episodes 50

    python scripts/compare_baselines.py --agent results/trained_agent/best.pt --baselines all --output results/
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baselines import (
    BaselineComparator,
    ComparisonConfig,
    LRUPolicy,
    AdaptiveLRUPolicy,
    LFUPolicy,
    WindowedLFUPolicy,
    StaticMarkovPolicy,
    InverseStaticMarkovPolicy,
    BalancedStaticMarkovPolicy,
    RandomPolicy,
    EpsilonRandomPolicy,
    AdaptivePolicy,
    MultiObjectiveAdaptivePolicy,
    OraclePolicy,
    RLAgentAdapter,
    TorchAgentAdapter
)
from src.integration.gym_environment import CachingEnv, CacheEnvConfig, SimulatorConfig
from src.cache.cache_manager import CacheManagerConfig
from src.rl.state import StateConfig
from src.rl.reward import RewardConfig
from src.rl.actions import ActionConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare baseline caching policies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all baselines
  python scripts/compare_baselines.py --baselines all --episodes 100
  
  # Compare specific baselines
  python scripts/compare_baselines.py --baselines lru,lfu,static_markov --episodes 50
  
  # Include trained RL agent
  python scripts/compare_baselines.py --agent results/best_agent.zip --episodes 100
  
  # Use custom environment config
  python scripts/compare_baselines.py --env-config configs/eval.yaml --output results/eval/
        """
    )

    # Environment configuration
    parser.add_argument(
        '--env-config',
        type=str,
        default=None,
        help='Path to environment YAML config file'
    )
    parser.add_argument(
        '--num-apis',
        type=int,
        default=20,
        help='Number of API endpoints to simulate'
    )
    parser.add_argument(
        '--session-length',
        type=int,
        default=50,
        help='Average session length'
    )

    # Baselines selection
    parser.add_argument(
        '--baselines',
        type=str,
        default='all',
        help='Comma-separated list of baselines to compare, or "all" (default: all). '
             'Options: lru, adaptive_lru, lfu, windowed_lfu, static_markov, inverse_markov, '
             'balanced_markov, random, epsilon_random, adaptive, multi_objective, oracle'
    )

    # RL agent
    parser.add_argument(
        '--agent',
        type=str,
        default=None,
        help='Path to trained RL agent (.zip for SB3, .pt for PyTorch)'
    )
    parser.add_argument(
        '--agent-name',
        type=str,
        default='DQN',
        help='Name for the RL agent (default: DQN)'
    )
    parser.add_argument(
        '--agent-type',
        type=str,
        default='sb3',
        choices=['sb3', 'torch'],
        help='Type of agent: sb3 (Stable-Baselines3) or torch (PyTorch)'
    )

    # Evaluation parameters
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes per policy (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='results/baseline_comparison',
        help='Output directory for results (default: results/baseline_comparison)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save detailed results to JSON'
    )

    # Misc
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )

    return parser.parse_args()


def load_env_config(config_path: str) -> CacheEnvConfig:
    """Load environment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Convert dict to config objects
    # This is a simplified version - you may need to adjust based on your YAML structure
    return CacheEnvConfig(
        simulator_config=SimulatorConfig(**config_dict.get('simulator', {})),
        cache_config=CacheManagerConfig(**config_dict.get('cache', {})),
        state_config=StateConfig(**config_dict.get('state', {})),
        reward_config=RewardConfig(**config_dict.get('reward', {})),
        action_config=ActionConfig(**config_dict.get('action', {})),
        max_steps_per_episode=config_dict.get('max_steps', 1000),
        seed=config_dict.get('seed', 42)
    )


def create_default_env_config(args) -> CacheEnvConfig:
    """Create default environment configuration from arguments."""
    return CacheEnvConfig(
        simulator_config=SimulatorConfig(
            num_apis=args.num_apis,
            session_length_range=(10, args.session_length),
            cascade_threshold=0.75
        ),
        max_steps_per_episode=500,
        use_real_services=False,
        episode_end_on_cascade=True,
        log_episode_metrics=False,
        seed=args.seed
    )


def get_baseline_policies(baseline_names: str) -> dict:
    """Get dictionary of baseline policies based on names."""
    all_baselines = {
        'lru': ('LRU', LRUPolicy()),
        'adaptive_lru': ('Adaptive LRU', AdaptiveLRUPolicy()),
        'lfu': ('LFU', LFUPolicy()),
        'windowed_lfu': ('Windowed LFU', WindowedLFUPolicy()),
        'static_markov': ('Static Markov', StaticMarkovPolicy()),
        'inverse_markov': ('Inverse Static Markov', InverseStaticMarkovPolicy()),
        'balanced_markov': ('Balanced Static Markov', BalancedStaticMarkovPolicy()),
        'random': ('Random', RandomPolicy()),
        'epsilon_random': ('ε-Random', EpsilonRandomPolicy(LRUPolicy(), epsilon=0.1)),
        'adaptive': ('Adaptive Heuristic', AdaptivePolicy()),
        'multi_objective': ('Multi-Objective Adaptive', MultiObjectiveAdaptivePolicy()),
        'oracle': ('Oracle', OraclePolicy(lookahead_window=10))
    }

    if baseline_names.lower() == 'all':
        # Exclude oracle by default (requires special handling)
        selected = {k: v for k, v in all_baselines.items() if k != 'oracle'}
    else:
        names = [name.strip().lower() for name in baseline_names.split(',')]
        selected = {}
        for name in names:
            if name in all_baselines:
                selected[name] = all_baselines[name]
            else:
                logger.warning(f"Unknown baseline: {name}")

    return selected


def load_agent(agent_path: str, agent_type: str, agent_name: str):
    """Load trained RL agent."""
    agent_path = Path(agent_path)

    if not agent_path.exists():
        raise FileNotFoundError(f"Agent not found: {agent_path}")

    if agent_type == 'sb3':
        # Load Stable-Baselines3 agent
        try:
            from stable_baselines3 import DQN, PPO, A2C

            # Try to determine algorithm from filename
            filename = agent_path.stem.lower()
            if 'dqn' in filename:
                agent = DQN.load(str(agent_path))
            elif 'ppo' in filename:
                agent = PPO.load(str(agent_path))
            elif 'a2c' in filename:
                agent = A2C.load(str(agent_path))
            else:
                # Default to DQN
                agent = DQN.load(str(agent_path))

            logger.info(f"Loaded SB3 agent: {agent_path}")
            return RLAgentAdapter(agent, agent_name)

        except Exception as e:
            logger.error(f"Failed to load SB3 agent: {e}")
            raise

    elif agent_type == 'torch':
        # Load PyTorch agent
        try:
            import torch
            from src.rl.agents.dqn_agent import DQNAgent

            # Create agent (you may need to adjust dimensions)
            agent = DQNAgent(state_dim=60, action_dim=7)

            # Load checkpoint
            checkpoint = torch.load(agent_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                agent.q_network.load_state_dict(checkpoint['model_state_dict'])
            else:
                agent.q_network.load_state_dict(checkpoint)

            logger.info(f"Loaded PyTorch agent: {agent_path}")
            return TorchAgentAdapter(agent, agent_name)

        except Exception as e:
            logger.error(f"Failed to load PyTorch agent: {e}")
            raise

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    """Main function."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    logger.info("="*80)
    logger.info("BASELINE CACHING POLICY COMPARISON")
    logger.info("="*80)

    # Create environment
    if args.env_config:
        logger.info(f"Loading environment config: {args.env_config}")
        env_config = load_env_config(args.env_config)
    else:
        logger.info("Using default environment configuration")
        env_config = create_default_env_config(args)

    env = CachingEnv(env_config)
    logger.info(f"Environment created: {args.episodes} episodes per policy")

    # Create comparator
    comparison_config = ComparisonConfig(
        num_episodes=args.episodes,
        seed=args.seed,
        track_detailed_metrics=True,
        save_episode_data=args.save_json
    )
    comparator = BaselineComparator(comparison_config)

    # Add baseline policies
    logger.info("\nRegistering baseline policies...")
    baselines = get_baseline_policies(args.baselines)
    for name, (display_name, policy) in baselines.items():
        comparator.add_policy(display_name, policy)
        logger.info(f"  - {display_name}")

    # Add trained agent if provided
    if args.agent:
        logger.info(f"\nLoading trained agent: {args.agent}")
        try:
            agent_policy = load_agent(args.agent, args.agent_type, args.agent_name)
            comparator.add_policy(args.agent_name, agent_policy)
            logger.info(f"  - {args.agent_name} (trained RL agent)")
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")
            logger.warning("Continuing without RL agent...")

    # Run comparison
    logger.info("\n" + "="*80)
    logger.info("RUNNING COMPARISON")
    logger.info("="*80)

    results = comparator.run_comparison(
        env,
        num_episodes=args.episodes,
        seed=args.seed,
        verbose=not args.quiet
    )

    # Display results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    print("\n" + results.to_string(index=False))

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results to CSV
    csv_path = output_path / 'results.csv'
    results.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to: {csv_path}")

    # Save detailed JSON if requested
    if args.save_json:
        json_path = output_path / 'detailed_results.json'
        comparator.save_results(json_path)
        logger.info(f"Detailed results saved to: {json_path}")

    # Generate report
    logger.info("\nGenerating comparison report...")
    report_path = comparator.generate_report(
        results,
        output_path,
        include_plots=not args.no_plots
    )
    logger.info(f"Report saved to: {report_path}")

    # Generate plots if not disabled
    if not args.no_plots:
        logger.info("\nGenerating comparison plots...")
        comparator.plot_comparison(
            metric='reward',
            output_path=str(output_path / 'reward_comparison.png')
        )
        comparator.plot_comparison(
            metric='cache_hit_rate',
            output_path=str(output_path / 'hitrate_comparison.png')
        )
        comparator.plot_comparison(
            metric='cascade_rate',
            output_path=str(output_path / 'cascade_comparison.png')
        )
        logger.info(f"Plots saved to: {output_path}")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    best_policy = results.iloc[0]
    logger.info(f"\n🏆 Best Policy: {best_policy['policy_name']}")
    logger.info(f"   Mean Reward: {best_policy['mean_reward']:.2f} ± {best_policy['std_reward']:.2f}")
    logger.info(f"   Cache Hit Rate: {best_policy['mean_hit_rate']:.2%}")
    logger.info(f"   Cascade Rate: {best_policy['cascade_rate']:.2%}")

    # Improvement over random baseline
    if 'Random' in results['policy_name'].values:
        random_reward = results[results['policy_name'] == 'Random']['mean_reward'].iloc[0]
        improvement = ((best_policy['mean_reward'] - random_reward) / abs(random_reward)) * 100
        logger.info(f"   Improvement over Random: {improvement:+.1f}%")

    logger.info("\n" + "="*80)
    logger.info("Comparison complete!")
    logger.info(f"All results saved to: {output_path}")
    logger.info("="*80)

    env.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)

