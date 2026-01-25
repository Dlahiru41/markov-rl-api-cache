"""Command-line interface for the preprocessing pipeline.

This CLI provides an easy-to-use interface for preprocessing API trace data.

Examples:
    # Process real data
    python scripts/preprocess.py -i data/raw/api_logs.csv -o data/processed

    # Generate and process synthetic data
    python scripts/preprocess.py --synthetic --num-users 1000 -o data/synthetic

    # Just validate existing data
    python scripts/preprocess.py -i data/raw/api_logs.csv --validate-only

    # Use custom config
    python scripts/preprocess.py --synthetic --config configs/custom.yaml -o data/test
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import json
import yaml

import click

# Add parent directory to path to import preprocessing modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.pipeline import PreprocessingPipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


@click.command()
@click.option(
    '--input', '-i',
    type=click.Path(exists=True, path_type=Path),
    help='Path to input data file (CSV, JSON, or Parquet)'
)
@click.option(
    '--output', '-o',
    type=click.Path(path_type=Path),
    default='data/processed',
    show_default=True,
    help='Output directory for processed artifacts'
)
@click.option(
    '--synthetic',
    is_flag=True,
    help='Generate synthetic data instead of loading from file'
)
@click.option(
    '--num-users',
    type=int,
    default=1000,
    show_default=True,
    help='Number of users for synthetic generation'
)
@click.option(
    '--sessions-per-user',
    type=int,
    default=5,
    show_default=True,
    help='Average sessions per user for synthetic generation'
)
@click.option(
    '--seed',
    type=int,
    default=42,
    show_default=True,
    help='Random seed for reproducibility'
)
@click.option(
    '--validate-only',
    is_flag=True,
    help='Just validate the input without full processing'
)
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Path to custom config file (YAML)'
)
@click.option(
    '--train-ratio',
    type=float,
    default=0.8,
    show_default=True,
    help='Fraction of data for training (0.0 to 1.0)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
def preprocess(
    input: Optional[Path],
    output: Path,
    synthetic: bool,
    num_users: int,
    sessions_per_user: int,
    seed: int,
    validate_only: bool,
    config: Optional[Path],
    train_ratio: float,
    verbose: bool
):
    """Preprocess API trace data for Markov chain and RL training.

    This tool orchestrates the complete preprocessing pipeline:
    - Load raw data or generate synthetic data
    - Validate data quality
    - Extract sessions and sequences
    - Fit feature engineer
    - Split into train/test sets
    - Save all processed artifacts

    You must specify either --input or --synthetic.
    """
    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not input and not synthetic:
        click.echo(
            click.style("Error: ", fg='red', bold=True) +
            "Must specify either --input or --synthetic"
        )
        sys.exit(1)

    if input and synthetic:
        click.echo(
            click.style("Error: ", fg='red', bold=True) +
            "Cannot specify both --input and --synthetic"
        )
        sys.exit(1)

    # Load config if provided
    config_dict = {}
    if config:
        click.echo(f"Loading config from {config}...")
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f)

    # Print banner
    click.echo()
    click.echo(click.style("="*70, fg='cyan'))
    click.echo(click.style("  Preprocessing Pipeline", fg='cyan', bold=True))
    click.echo(click.style("="*70, fg='cyan'))
    click.echo()

    # Show configuration
    if synthetic:
        click.echo(click.style("Mode: ", fg='yellow') + "Synthetic data generation")
        click.echo(f"  • Users: {num_users:,}")
        click.echo(f"  • Sessions per user: {sessions_per_user}")
    else:
        click.echo(click.style("Mode: ", fg='yellow') + "Load from file")
        click.echo(f"  • Input: {input}")

    click.echo(f"  • Output: {output}")
    click.echo(f"  • Seed: {seed}")
    if not validate_only:
        click.echo(f"  • Train ratio: {train_ratio:.1%}")
    click.echo()

    try:
        # Initialize pipeline
        pipeline = PreprocessingPipeline(
            output_dir=str(output),
            train_ratio=train_ratio,
            seed=seed,
            config=config_dict
        )

        # Run pipeline
        results = pipeline.run(
            input_path=str(input) if input else None,
            generate_synthetic=synthetic,
            num_users=num_users,
            sessions_per_user=(sessions_per_user, 2),  # (mean, std)
            validate_only=validate_only
        )

        # Show results
        click.echo()
        click.echo(click.style("="*70, fg='green'))
        click.echo(click.style("  [OK] Success!", fg='green', bold=True))
        click.echo(click.style("="*70, fg='green'))
        click.echo()

        if validate_only:
            # Show validation results
            click.echo(click.style("Validation Results:", fg='yellow', bold=True))
            click.echo(f"  • Valid: {click.style('YES', fg='green') if results['valid'] else click.style('NO', fg='red')}")
            click.echo(f"  • Total sessions: {results['total_sessions']:,}")
            click.echo(f"  • Total calls: {results['total_calls']:,}")
            click.echo(f"  • Errors: {len(results['errors'])}")
            click.echo(f"  • Warnings: {len(results['warnings'])}")
            click.echo(f"  • Anomalies: {len(results['anomalies'])}")

            if results['errors'] and len(results['errors']) <= 5:
                click.echo()
                click.echo(click.style("First few errors:", fg='red'))
                for error in results['errors'][:5]:
                    click.echo(f"  • {error}")

            if results['quality_metrics']:
                click.echo()
                click.echo(click.style("Quality Metrics:", fg='yellow'))
                for metric, value in results['quality_metrics'].items():
                    click.echo(f"  • {metric}: {value:.1%}")
        else:
            # Show output files
            click.echo(click.style("Output Files:", fg='yellow', bold=True))
            for name, path in results.items():
                click.echo(f"  • {name}: {path}")

            # Show report path prominently
            if 'report' in results:
                click.echo()
                click.echo(
                    click.style("📄 View the full report: ", fg='cyan', bold=True) +
                    click.style(results['report'], fg='cyan')
                )

            # Show quick stats
            if 'statistics' in results:
                stats_path = Path(results['statistics'])
                if stats_path.exists():
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)

                    click.echo()
                    click.echo(click.style("Quick Statistics:", fg='yellow', bold=True))
                    click.echo(f"  • Raw sessions: {stats['raw_sessions']:,}")
                    click.echo(f"  • Raw calls: {stats['raw_calls']:,}")
                    click.echo(f"  • Train sessions: {stats['train_sessions']:,}")
                    click.echo(f"  • Test sessions: {stats['test_sessions']:,}")
                    click.echo(f"  • Sequences: {stats['sequences_extracted']:,}")
                    click.echo(f"  • Feature dimension: {stats['feature_dimension']}")

                    duration = stats.get('duration_seconds')
                    if duration is not None:
                        click.echo(f"  • Duration: {duration:.1f}s")

        click.echo()
        sys.exit(0)

    except Exception as e:
        click.echo()
        click.echo(click.style("="*70, fg='red'))
        click.echo(click.style("  [FAIL] Error!", fg='red', bold=True))
        click.echo(click.style("="*70, fg='red'))
        click.echo()
        click.echo(click.style(f"Error: {str(e)}", fg='red'))

        if verbose:
            click.echo()
            click.echo(click.style("Traceback:", fg='yellow'))
            import traceback
            traceback.print_exc()

        click.echo()
        sys.exit(1)


if __name__ == '__main__':
    preprocess()

