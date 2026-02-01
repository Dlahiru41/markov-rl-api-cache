"""
Demonstration script for the results analysis and visualization module.

Shows typical usage patterns for analyzing experiment results,
performing statistical tests, and generating thesis-quality figures.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation.analyzer import ResultsAnalyzer, ResultsVisualizer
from evaluation.report_generator import ReportGenerator


def demo_basic_analysis():
    """Demonstrate basic analysis workflow."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Results Analysis")
    print("="*80)

    # Load results
    analyzer = ResultsAnalyzer('results/experiments')
    print(f"✓ Loaded {len(analyzer.experiments)} experiments")

    # List available experiments
    print("\nAvailable experiments:")
    for exp_id in list(analyzer.experiments.keys())[:5]:
        results = analyzer.load_experiment(exp_id)
        if results:
            print(f"  - {exp_id}: {results[0].config.name} ({len(results)} seeds)")

    # Summarize a specific experiment
    if analyzer.experiments:
        exp_id = list(analyzer.experiments.keys())[0]
        summary = analyzer.summarize_experiment(exp_id)

        print(f"\nSummary for {summary['name']}:")
        print(f"  Seeds: {summary['num_completed']}/{summary['num_seeds']}")
        print(f"\n  Metrics:")
        for metric, stats in summary['metrics'].items():
            print(f"    {metric:20s}: {stats['mean']:.3f} ± {stats['std']:.3f}")


def demo_statistical_comparison():
    """Demonstrate statistical comparison between methods."""
    print("\n" + "="*80)
    print("DEMO 2: Statistical Comparison")
    print("="*80)

    analyzer = ResultsAnalyzer('results/experiments')

    # Get two experiments to compare
    exp_ids = list(analyzer.experiments.keys())
    if len(exp_ids) < 2:
        print("Need at least 2 experiments for comparison")
        return

    # Load results
    exp1_results = analyzer.load_experiment(exp_ids[0])
    exp2_results = analyzer.load_experiment(exp_ids[1])

    if not exp1_results or not exp2_results:
        print("Could not load experiment results")
        return

    # Extract metric values
    metric = 'mean_reward'
    values1 = [r.final_eval_metrics.get(metric, 0) for r in exp1_results if r.status == 'completed']
    values2 = [r.final_eval_metrics.get(metric, 0) for r in exp2_results if r.status == 'completed']

    # Compare
    comparison = analyzer.compare_two_methods(values1, values2, metric=metric)

    print(f"\nComparing: {exp1_results[0].config.name} vs {exp2_results[0].config.name}")
    print(f"Metric: {metric}")
    print(f"\nMethod A ({exp1_results[0].config.name}):")
    print(f"  Mean: {comparison['mean_a']:.4f}")
    print(f"  Std:  {comparison['std_a']:.4f}")
    print(f"  95% CI: [{comparison['ci_a'][0]:.4f}, {comparison['ci_a'][1]:.4f}]")
    print(f"\nMethod B ({exp2_results[0].config.name}):")
    print(f"  Mean: {comparison['mean_b']:.4f}")
    print(f"  Std:  {comparison['std_b']:.4f}")
    print(f"  95% CI: [{comparison['ci_b'][0]:.4f}, {comparison['ci_b'][1]:.4f}]")
    print(f"\nStatistical Tests:")
    print(f"  t-test p-value:         {comparison['t_test_pvalue']:.4f}")
    print(f"  Mann-Whitney U p-value: {comparison['mannwhitneyu_pvalue']:.4f}")
    print(f"  Cohen's d (effect size): {comparison['cohens_d']:.2f}")
    print(f"  Significant (α=0.05):   {'Yes ✓' if comparison['significant'] else 'No ✗'}")
    print(f"\n  {comparison['interpretation']}")


def demo_multi_method_comparison():
    """Demonstrate comparison of multiple methods."""
    print("\n" + "="*80)
    print("DEMO 3: Multi-Method Comparison")
    print("="*80)

    analyzer = ResultsAnalyzer('results/experiments')

    # Get multiple experiments
    exp_ids = list(analyzer.experiments.keys())[:3]
    if len(exp_ids) < 3:
        print("Need at least 3 experiments for multi-method comparison")
        return

    # Collect values
    metric = 'mean_reward'
    results_dict = {}

    for exp_id in exp_ids:
        results = analyzer.load_experiment(exp_id)
        if results:
            values = [r.final_eval_metrics.get(metric, 0) for r in results if r.status == 'completed']
            if values:
                results_dict[results[0].config.name] = values

    if len(results_dict) < 3:
        print("Not enough valid data")
        return

    # Compare
    comparison = analyzer.compare_multiple_methods(results_dict, metric=metric)

    print(f"\nComparing {len(results_dict)} methods on {metric}")
    print(f"\nOverall Tests:")
    print(f"  ANOVA p-value:          {comparison['anova_pvalue']:.4f}")
    print(f"  Kruskal-Wallis p-value: {comparison['kruskal_pvalue']:.4f}")
    print(f"  Significant (α=0.05):   {'Yes ✓' if comparison['significant'] else 'No ✗'}")
    print(f"\nRankings:")
    for rank, (name, mean) in enumerate(comparison['rankings'], 1):
        print(f"  {rank}. {name:30s}: {mean:.4f}")

    # Pairwise comparisons if available
    if 'pairwise_comparisons' in comparison:
        print(f"\nPairwise Comparisons (Tukey HSD):")
        for pw in comparison['pairwise_comparisons']:
            sig = " ***" if pw['reject'] else "    "
            print(f"{sig} {pw['group1']:20s} vs {pw['group2']:20s}: "
                  f"diff={pw['meandiff']:7.2f}, p={pw['pvalue']:.4f}")


def demo_learning_curve_analysis():
    """Demonstrate learning curve analysis."""
    print("\n" + "="*80)
    print("DEMO 4: Learning Curve Analysis")
    print("="*80)

    analyzer = ResultsAnalyzer('results/experiments')

    # Find an experiment with training history
    exp_id = None
    for eid in analyzer.experiments.keys():
        results = analyzer.load_experiment(eid)
        if results and results[0].training_history:
            exp_id = eid
            break

    if not exp_id:
        print("No experiments with training history found")
        return

    # Analyze learning
    analysis = analyzer.analyze_learning_curve(exp_id)

    print(f"\nLearning Analysis for {analysis['experiment_id']}:")
    print(f"  Total Episodes: {analysis['total_episodes']}")
    if analysis['convergence_episode']:
        print(f"  Convergence Episode: {analysis['convergence_episode']}")
    print(f"  Final Performance: {analysis['final_performance']:.2f} ± {analysis['final_std']:.2f}")
    print(f"  Stability (CV): {analysis['stability_cv']:.4f}")
    print(f"  Trend: {analysis['trend']}")
    print(f"  Trend Slope: {analysis['trend_slope']:.4f}")
    print(f"  R²: {analysis['trend_r_squared']:.4f}")


def demo_visualizations():
    """Demonstrate visualization generation."""
    print("\n" + "="*80)
    print("DEMO 5: Generating Visualizations")
    print("="*80)

    analyzer = ResultsAnalyzer('results/experiments')
    visualizer = ResultsVisualizer()

    exp_ids = list(analyzer.experiments.keys())[:3]
    if len(exp_ids) < 2:
        print("Need at least 2 experiments for visualizations")
        return

    output_dir = Path('results/demo_figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating figures in {output_dir}/...")

    # 1. Learning curves
    try:
        lc_path = output_dir / 'learning_curves.pdf'
        visualizer.plot_learning_curves(
            analyzer, exp_ids[:2],
            metric='reward',
            output_path=str(lc_path),
            title='Training Progress Comparison'
        )
        print(f"  ✓ Learning curves: {lc_path}")
    except Exception as e:
        print(f"  ✗ Learning curves failed: {e}")

    # 2. Bar comparison
    try:
        bar_path = output_dir / 'bar_comparison.pdf'
        visualizer.plot_bar_comparison(
            analyzer, exp_ids[:3],
            metric='mean_reward',
            output_path=str(bar_path),
            title='Performance Comparison'
        )
        print(f"  ✓ Bar comparison: {bar_path}")
    except Exception as e:
        print(f"  ✗ Bar comparison failed: {e}")

    # 3. Box plot
    try:
        box_path = output_dir / 'box_plot.pdf'
        visualizer.plot_box_comparison(
            analyzer, exp_ids[:3],
            metric='mean_reward',
            output_path=str(box_path),
            title='Reward Distribution'
        )
        print(f"  ✓ Box plot: {box_path}")
    except Exception as e:
        print(f"  ✗ Box plot failed: {e}")

    # 4. Radar chart
    try:
        radar_path = output_dir / 'radar_chart.pdf'
        visualizer.plot_radar_chart(
            analyzer, exp_ids[:3],
            metrics=['mean_reward', 'cache_hit_rate'],
            output_path=str(radar_path),
            title='Multi-Metric Comparison'
        )
        print(f"  ✓ Radar chart: {radar_path}")
    except Exception as e:
        print(f"  ✗ Radar chart failed: {e}")


def demo_report_generation():
    """Demonstrate report generation."""
    print("\n" + "="*80)
    print("DEMO 6: Generating Reports")
    print("="*80)

    analyzer = ResultsAnalyzer('results/experiments')
    visualizer = ResultsVisualizer()
    generator = ReportGenerator(analyzer, visualizer)

    exp_ids = list(analyzer.experiments.keys())[:3]
    if not exp_ids:
        print("No experiments found")
        return

    output_dir = Path('results/demo_reports')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating reports in {output_dir}/...")

    # 1. Single experiment report (HTML)
    try:
        report_path = output_dir / 'experiment_report.html'
        generator.generate_experiment_report(
            exp_ids[0],
            str(report_path),
            format='html',
            include_plots=False
        )
        print(f"  ✓ Experiment report: {report_path}")
    except Exception as e:
        print(f"  ✗ Experiment report failed: {e}")

    # 2. Comparison report (HTML)
    if len(exp_ids) >= 2:
        try:
            comp_path = output_dir / 'comparison_report.html'
            generator.generate_comparison_report(
                exp_ids[:3],
                str(comp_path),
                format='html',
                metrics=['mean_reward', 'cache_hit_rate'],
                include_plots=False
            )
            print(f"  ✓ Comparison report: {comp_path}")
        except Exception as e:
            print(f"  ✗ Comparison report failed: {e}")

    # 3. Thesis chapter (LaTeX)
    try:
        thesis_path = output_dir / 'results_chapter.tex'
        generator.generate_thesis_chapter(
            'results/experiments',
            str(thesis_path),
            chapter_title='Experimental Results'
        )
        print(f"  ✓ Thesis chapter: {thesis_path}")
    except Exception as e:
        print(f"  ✗ Thesis chapter failed: {e}")


def demo_complete_workflow():
    """Demonstrate complete analysis workflow."""
    print("\n" + "="*80)
    print("DEMO 7: Complete Workflow")
    print("="*80)

    print("\nTypical workflow for thesis evaluation:")
    print("-" * 80)

    # Step 1: Load results
    print("\n1. Load experiment results")
    analyzer = ResultsAnalyzer('results/experiments')
    print(f"   ✓ Loaded {len(analyzer.experiments)} experiments")

    exp_ids = list(analyzer.experiments.keys())
    if len(exp_ids) < 2:
        print("   Need at least 2 experiments")
        return

    # Step 2: Identify best configuration
    print("\n2. Identify best configuration")
    try:
        best = analyzer.identify_best_configuration(exp_ids[:5], metric='mean_reward')
        print(f"   ✓ Best: {best['best_name']} (mean={best['best_mean']:.2f})")
        if 'vs_second_best' in best:
            print(f"   ✓ Improvement over 2nd: {best['vs_second_best']['improvement']:.1f}%")
            print(f"   ✓ Statistically significant: {best['vs_second_best']['significant']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Step 3: Create summary table
    print("\n3. Create summary table")
    try:
        table = analyzer.create_summary_table(exp_ids[:3], ['mean_reward', 'cache_hit_rate'])
        print(f"   ✓ Created table: {table.shape[0]} experiments × {table.shape[1]} columns")
        print("\n   Preview:")
        print(table.to_string(index=False, max_rows=3, max_cols=5))
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Step 4: Generate all thesis figures
    print("\n4. Generate thesis figures")
    try:
        visualizer = ResultsVisualizer()
        figures = visualizer.generate_all_thesis_figures(
            analyzer,
            exp_ids[:3],
            'results/demo_thesis/figures',
            metrics=['mean_reward', 'cache_hit_rate']
        )
        print(f"   ✓ Generated {len(figures)} figures")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    # Step 5: Generate reports
    print("\n5. Generate reports")
    try:
        generator = ReportGenerator(analyzer, visualizer)

        # Comparison report
        generator.generate_comparison_report(
            exp_ids[:3],
            'results/demo_thesis/comparison_report.html',
            format='html',
            metrics=['mean_reward', 'cache_hit_rate'],
            include_plots=False
        )
        print(f"   ✓ Comparison report")

        # Thesis chapter
        generator.generate_thesis_chapter(
            'results/experiments',
            'results/demo_thesis/results_chapter.tex'
        )
        print(f"   ✓ Thesis chapter")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

    print("\n✅ Workflow complete!")


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print("RESULTS ANALYSIS MODULE DEMONSTRATION")
    print("="*80)
    print("\nThis script demonstrates the capabilities of the results analysis module.")
    print("Make sure you have experiment results in 'results/experiments/' directory.")
    print("\nNote: Some demos may skip if required data is not available.")

    demos = [
        ("Basic Analysis", demo_basic_analysis),
        ("Statistical Comparison", demo_statistical_comparison),
        ("Multi-Method Comparison", demo_multi_method_comparison),
        ("Learning Curve Analysis", demo_learning_curve_analysis),
        ("Visualizations", demo_visualizations),
        ("Report Generation", demo_report_generation),
        ("Complete Workflow", demo_complete_workflow),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ {name} demo failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nFor more information:")
    print("  - See validate_analyzer.py for comprehensive tests")
    print("  - Use scripts/analyze.py for command-line interface")
    print("  - Check evaluation/analyzer.py for API documentation")


if __name__ == '__main__':
    main()

