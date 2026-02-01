"""
Command-line interface for experiment results analysis.

Provides comprehensive tools for analyzing experiment results,
generating reports, and creating thesis-quality figures.

Usage:
    python scripts/analyze.py summarize --results-dir results/experiments
    python scripts/analyze.py compare --experiments exp1,exp2 --metric reward
    python scripts/analyze.py plot learning_curves --experiments exp1,exp2
    python scripts/analyze.py report --experiments exp1,exp2 --output reports/
    python scripts/analyze.py thesis --results-dir results/ --output thesis/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.analyzer import ResultsAnalyzer, ResultsVisualizer
from evaluation.report_generator import ReportGenerator


def cmd_summarize(args):
    """Summarize all experiments in directory."""
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}\n")

    analyzer = ResultsAnalyzer(args.results_dir)

    print(f"Results Directory: {args.results_dir}")
    print(f"Total Experiments: {len(analyzer.experiments)}")
    print("")

    # Group by tags
    all_tags = set()
    for exp_id, results in analyzer.experiments.items():
        if results:
            all_tags.update(results[0].config.tags)

    print(f"Tags Found: {', '.join(sorted(all_tags))}")
    print("")

    # List all experiments
    print("Experiments:")
    print("-" * 80)
    print(f"{'ID':<30} {'Name':<30} {'Seeds':<10} {'Status'}")
    print("-" * 80)

    for exp_id, results in sorted(analyzer.experiments.items()):
        if results:
            name = results[0].config.name[:28]
            num_seeds = len(results)
            completed = sum(1 for r in results if r.status == 'completed')
            status = f"{completed}/{num_seeds} completed"
            print(f"{exp_id:<30} {name:<30} {num_seeds:<10} {status}")

    print("-" * 80)
    print("")

    # Summary statistics
    if args.detailed:
        print("\nDetailed Statistics:")
        print("=" * 80)
        for exp_id in list(analyzer.experiments.keys())[:args.limit]:
            try:
                summary = analyzer.summarize_experiment(exp_id)
                print(f"\n{summary['name']}:")
                for metric, stats in summary['metrics'].items():
                    print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")


def cmd_compare(args):
    """Compare multiple experiments statistically."""
    print(f"\n{'='*80}")
    print("STATISTICAL COMPARISON")
    print(f"{'='*80}\n")

    analyzer = ResultsAnalyzer(args.results_dir)

    # Parse experiment IDs
    experiment_ids = args.experiments.split(',')
    print(f"Comparing {len(experiment_ids)} experiments on metric: {args.metric}")
    print("")

    # Collect values for each experiment
    method_values = {}
    for exp_id in experiment_ids:
        results = analyzer.load_experiment(exp_id)
        if results:
            completed = [r for r in results if r.status == 'completed']
            if completed:
                values = [r.final_eval_metrics.get(args.metric, float('nan')) for r in completed]
                values = [v for v in values if not float('isnan' if isinstance(v, str) else 'nan')(v)]
                if values:
                    method_values[completed[0].config.name] = values

    if len(method_values) < 2:
        print("Error: Need at least 2 experiments with valid data")
        return

    # Two-way comparison
    if len(method_values) == 2:
        names = list(method_values.keys())
        comparison = analyzer.compare_two_methods(
            method_values[names[0]],
            method_values[names[1]],
            metric=args.metric
        )

        print(f"Comparing: {names[0]} vs {names[1]}")
        print("-" * 80)
        print(f"Method A ({names[0]}):")
        print(f"  Mean: {comparison['mean_a']:.4f}")
        print(f"  Std:  {comparison['std_a']:.4f}")
        print(f"  95% CI: [{comparison['ci_a'][0]:.4f}, {comparison['ci_a'][1]:.4f}]")
        print("")
        print(f"Method B ({names[1]}):")
        print(f"  Mean: {comparison['mean_b']:.4f}")
        print(f"  Std:  {comparison['std_b']:.4f}")
        print(f"  95% CI: [{comparison['ci_b'][0]:.4f}, {comparison['ci_b'][1]:.4f}]")
        print("")
        print("Statistical Tests:")
        print(f"  t-test p-value:         {comparison['t_test_pvalue']:.4f}")
        print(f"  Mann-Whitney U p-value: {comparison['mannwhitneyu_pvalue']:.4f}")
        print(f"  Cohen's d:              {comparison['cohens_d']:.2f}")
        print(f"  Significant (α=0.05):   {comparison['significant']}")
        print("")
        print(f"Interpretation: {comparison['interpretation']}")

    # Multi-way comparison
    else:
        comparison = analyzer.compare_multiple_methods(method_values, metric=args.metric)

        print("Overall Comparison:")
        print("-" * 80)
        print(f"ANOVA p-value:          {comparison['anova_pvalue']:.4f}")
        print(f"Kruskal-Wallis p-value: {comparison['kruskal_pvalue']:.4f}")
        print(f"Significant (α=0.05):   {comparison['significant']}")
        print("")
        print("Rankings:")
        for rank, (name, mean) in enumerate(comparison['rankings'], 1):
            print(f"  {rank}. {name}: {mean:.4f}")
        print("")

        if 'pairwise_comparisons' in comparison:
            print("Pairwise Comparisons (Tukey HSD):")
            print("-" * 80)
            for pw in comparison['pairwise_comparisons']:
                sig = "***" if pw['reject'] else "   "
                print(f"  {sig} {pw['group1']} vs {pw['group2']}: "
                      f"diff={pw['meandiff']:.4f}, p={pw['pvalue']:.4f}")

    print("")


def cmd_plot(args):
    """Generate specific plot."""
    print(f"\n{'='*80}")
    print(f"GENERATING PLOT: {args.plot_type}")
    print(f"{'='*80}\n")

    analyzer = ResultsAnalyzer(args.results_dir)
    visualizer = ResultsVisualizer(dpi=args.dpi)

    # Parse experiment IDs
    experiment_ids = args.experiments.split(',')

    # Output path
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.plot_type}.{args.format}"

    try:
        if args.plot_type == 'learning_curves':
            visualizer.plot_learning_curves(
                analyzer,
                experiment_ids,
                metric=args.metric,
                output_path=str(output_path),
                title=args.title
            )

        elif args.plot_type == 'bar_comparison':
            visualizer.plot_bar_comparison(
                analyzer,
                experiment_ids,
                metric=args.metric,
                output_path=str(output_path),
                title=args.title
            )

        elif args.plot_type == 'box_plot':
            visualizer.plot_box_comparison(
                analyzer,
                experiment_ids,
                metric=args.metric,
                output_path=str(output_path),
                title=args.title
            )

        elif args.plot_type == 'radar_chart':
            metrics = args.metrics.split(',') if args.metrics else ['mean_reward', 'cache_hit_rate']
            visualizer.plot_radar_chart(
                analyzer,
                experiment_ids,
                metrics=metrics,
                output_path=str(output_path),
                title=args.title
            )

        elif args.plot_type == 'training_comparison':
            metrics = args.metrics.split(',') if args.metrics else ['reward', 'loss', 'cache_hit_rate']
            visualizer.plot_training_comparison(
                analyzer,
                experiment_ids,
                metrics=metrics,
                output_path=str(output_path)
            )

        else:
            print(f"Error: Unknown plot type '{args.plot_type}'")
            print("Available types: learning_curves, bar_comparison, box_plot, radar_chart, training_comparison")
            return

        print(f"✓ Plot saved to: {output_path}")

    except Exception as e:
        print(f"Error generating plot: {e}")
        import traceback
        traceback.print_exc()


def cmd_report(args):
    """Generate comprehensive report."""
    print(f"\n{'='*80}")
    print("GENERATING REPORT")
    print(f"{'='*80}\n")

    analyzer = ResultsAnalyzer(args.results_dir)
    visualizer = ResultsVisualizer()
    generator = ReportGenerator(analyzer, visualizer)

    # Parse experiment IDs
    experiment_ids = args.experiments.split(',')

    # Output path
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(experiment_ids) == 1:
        # Single experiment report
        output_path = output_dir / f"report_{experiment_ids[0]}.{args.format}"
        generator.generate_experiment_report(
            experiment_ids[0],
            str(output_path),
            format=args.format
        )
    else:
        # Comparison report
        output_path = output_dir / f"comparison_report.{args.format}"
        metrics = args.metrics.split(',') if args.metrics else ['mean_reward', 'cache_hit_rate']
        generator.generate_comparison_report(
            experiment_ids,
            str(output_path),
            format=args.format,
            metrics=metrics
        )

    print(f"\n✓ Report generated: {output_path}")


def cmd_thesis(args):
    """Generate all thesis materials."""
    print(f"\n{'='*80}")
    print("GENERATING THESIS MATERIALS")
    print(f"{'='*80}\n")

    analyzer = ResultsAnalyzer(args.results_dir)
    visualizer = ResultsVisualizer()
    generator = ReportGenerator(analyzer, visualizer)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    print("Generating figures...")
    experiment_ids = list(analyzer.experiments.keys())

    if args.experiments:
        experiment_ids = args.experiments.split(',')

    figures = visualizer.generate_all_thesis_figures(
        analyzer,
        experiment_ids,
        str(output_dir / 'figures')
    )

    # Generate thesis chapter
    print("\nGenerating thesis chapter...")
    chapter_path = output_dir / 'results_chapter.tex'
    generator.generate_thesis_chapter(
        args.results_dir,
        str(chapter_path)
    )

    # Generate comparison reports
    print("\nGenerating comparison reports...")

    # HTML report
    html_path = output_dir / 'comparison_report.html'
    metrics = ['mean_reward', 'cache_hit_rate', 'latency']
    generator.generate_comparison_report(
        experiment_ids,
        str(html_path),
        format='html',
        metrics=metrics
    )

    # LaTeX tables
    latex_path = output_dir / 'comparison_tables.tex'
    generator.generate_comparison_report(
        experiment_ids,
        str(latex_path),
        format='latex',
        metrics=metrics
    )

    print(f"\n{'='*80}")
    print("THESIS MATERIALS GENERATED")
    print(f"{'='*80}")
    print(f"Output Directory: {output_dir}")
    print(f"  - Figures: {len(figures)} files")
    print(f"  - Chapter: {chapter_path}")
    print(f"  - HTML Report: {html_path}")
    print(f"  - LaTeX Tables: {latex_path}")
    print("")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Experiment Results Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize all experiments
  python scripts/analyze.py summarize --results-dir results/experiments
  
  # Compare two experiments
  python scripts/analyze.py compare --experiments dqn_baseline,lru_baseline --metric mean_reward
  
  # Generate learning curves
  python scripts/analyze.py plot learning_curves --experiments dqn_baseline,dqn_tuned --output figures/
  
  # Generate comparison report
  python scripts/analyze.py report --experiments dqn,lru,lfu --output reports/ --format html
  
  # Generate all thesis materials
  python scripts/analyze.py thesis --results-dir results/experiments --output thesis/generated/
        """
    )

    # Global arguments
    parser.add_argument('--results-dir', default='results/experiments',
                       help='Directory containing experiment results')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Summarize command
    summarize_parser = subparsers.add_parser('summarize', help='Summarize all experiments')
    summarize_parser.add_argument('--detailed', action='store_true',
                                 help='Show detailed statistics')
    summarize_parser.add_argument('--limit', type=int, default=10,
                                 help='Limit number of detailed summaries')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare experiments statistically')
    compare_parser.add_argument('--experiments', required=True,
                               help='Comma-separated experiment IDs')
    compare_parser.add_argument('--metric', default='mean_reward',
                               help='Metric to compare')

    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Generate specific plot')
    plot_parser.add_argument('plot_type',
                            choices=['learning_curves', 'bar_comparison', 'box_plot',
                                   'radar_chart', 'training_comparison'],
                            help='Type of plot to generate')
    plot_parser.add_argument('--experiments', required=True,
                            help='Comma-separated experiment IDs')
    plot_parser.add_argument('--metric', default='reward',
                            help='Primary metric to plot')
    plot_parser.add_argument('--metrics',
                            help='Comma-separated metrics (for multi-metric plots)')
    plot_parser.add_argument('--output', default='figures/',
                            help='Output directory')
    plot_parser.add_argument('--format', default='pdf',
                            choices=['pdf', 'png', 'svg', 'eps'],
                            help='Output format')
    plot_parser.add_argument('--dpi', type=int, default=300,
                            help='DPI for raster formats')
    plot_parser.add_argument('--title',
                            help='Custom plot title')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate comprehensive report')
    report_parser.add_argument('--experiments', required=True,
                              help='Comma-separated experiment IDs')
    report_parser.add_argument('--output', default='reports/',
                              help='Output directory')
    report_parser.add_argument('--format', default='html',
                              choices=['markdown', 'html', 'latex'],
                              help='Report format')
    report_parser.add_argument('--metrics',
                              help='Comma-separated metrics to include')

    # Thesis command
    thesis_parser = subparsers.add_parser('thesis', help='Generate all thesis materials')
    thesis_parser.add_argument('--output', default='thesis/generated/',
                              help='Output directory')
    thesis_parser.add_argument('--experiments',
                              help='Comma-separated experiment IDs (default: all)')

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == 'summarize':
        cmd_summarize(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'plot':
        cmd_plot(args)
    elif args.command == 'report':
        cmd_report(args)
    elif args.command == 'thesis':
        cmd_thesis(args)


if __name__ == '__main__':
    main()

