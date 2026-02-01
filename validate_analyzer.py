"""
Validation script for the results analysis and visualization module.

Tests all functionality of the analyzer, visualizer, and report generator
to ensure proper operation.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from evaluation.analyzer import ResultsAnalyzer, ResultsVisualizer
from evaluation.report_generator import ReportGenerator
from evaluation.experiment_runner import ExperimentResult, ExperimentConfig


def create_mock_results(output_dir: str, num_experiments: int = 3):
    """Create mock experiment results for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / 'results').mkdir(exist_ok=True)
    (output_path / 'configs').mkdir(exist_ok=True)

    all_results = {}

    # Create mock experiments
    experiment_names = ['dqn_baseline', 'lru_baseline', 'lfu_baseline']

    for exp_idx, exp_name in enumerate(experiment_names[:num_experiments]):
        exp_id = f"{exp_name}_{exp_idx:04d}"

        # Create config
        config = ExperimentConfig(
            name=exp_name,
            description=f"Test experiment {exp_name}",
            hypothesis="Test hypothesis",
            controller_config={'learning_rate': 0.001 * (exp_idx + 1)},
            num_training_episodes=100,
            num_eval_episodes=20,
            seeds=[42, 123, 456],
            tags=['test', 'baseline' if 'baseline' in exp_name else 'rl']
        )

        # Create results for multiple seeds
        seed_results = []
        for seed in config.seeds:
            # Generate training history
            training_history = []
            base_reward = 800 + exp_idx * 100  # Different performance for each method
            for ep in range(0, 101, 10):
                reward = base_reward + np.random.randn() * 50 + ep * 2  # Improving over time
                loss = 0.5 * np.exp(-ep / 50) + np.random.rand() * 0.1  # Decreasing loss
                epsilon = max(0.01, 1.0 - ep / 100)

                training_history.append((ep, {
                    'reward': reward,
                    'loss': loss,
                    'epsilon': epsilon,
                    'cache_hit_rate': 0.6 + np.random.rand() * 0.2
                }))

            # Final evaluation metrics
            final_eval_metrics = {
                'mean_reward': base_reward + 200 + np.random.randn() * 20,
                'std_reward': 15 + np.random.rand() * 5,
                'cache_hit_rate': 0.7 + np.random.rand() * 0.15,
                'cascade_rate': 0.05 + np.random.rand() * 0.03,
                'latency': 100 + np.random.rand() * 20
            }

            result = ExperimentResult(
                config=config,
                seed=seed,
                training_history=training_history,
                final_eval_metrics=final_eval_metrics,
                training_time_seconds=300 + np.random.rand() * 100,
                peak_memory_mb=1024 + np.random.rand() * 512,
                status='completed',
                timestamp='2024-01-01T12:00:00'
            )

            seed_results.append(result)

        all_results[exp_id] = seed_results

    # Save consolidated results
    with open(output_path / 'all_results.json', 'w') as f:
        data = {}
        for exp_id, results in all_results.items():
            data[exp_id] = [r.to_dict() for r in results]
        json.dump(data, f, indent=2)

    print(f"✓ Created {num_experiments} mock experiments in {output_dir}")
    return output_path


def test_analyzer():
    """Test ResultsAnalyzer functionality."""
    print("\n" + "="*80)
    print("TEST 1: ResultsAnalyzer")
    print("="*80)

    # Create mock data
    test_dir = create_mock_results('results/test_analyzer', num_experiments=3)

    try:
        # Initialize analyzer
        analyzer = ResultsAnalyzer(str(test_dir))
        print(f"✓ Loaded {len(analyzer.experiments)} experiments")

        # Test load_experiment
        exp_ids = list(analyzer.experiments.keys())
        if exp_ids:
            exp_id = exp_ids[0]
            results = analyzer.load_experiment(exp_id)
            print(f"✓ Loaded experiment {exp_id}: {len(results)} seeds")

        # Test summarize_experiment
        summary = analyzer.summarize_experiment(exp_id)
        print(f"✓ Summarized experiment: {len(summary['metrics'])} metrics")

        # Test compare_two_methods
        if len(exp_ids) >= 2:
            exp1_results = analyzer.load_experiment(exp_ids[0])
            exp2_results = analyzer.load_experiment(exp_ids[1])

            values1 = [r.final_eval_metrics['mean_reward'] for r in exp1_results]
            values2 = [r.final_eval_metrics['mean_reward'] for r in exp2_results]

            comparison = analyzer.compare_two_methods(values1, values2, metric='mean_reward')
            print(f"✓ Two-method comparison: p-value={comparison['t_test_pvalue']:.4f}, "
                  f"significant={comparison['significant']}")

        # Test compare_multiple_methods
        if len(exp_ids) >= 3:
            results_dict = {}
            for exp_id in exp_ids[:3]:
                results = analyzer.load_experiment(exp_id)
                values = [r.final_eval_metrics['mean_reward'] for r in results]
                results_dict[exp_id] = values

            multi_comparison = analyzer.compare_multiple_methods(results_dict)
            print(f"✓ Multi-method comparison: ANOVA p-value={multi_comparison['anova_pvalue']:.4f}")

        # Test confidence intervals
        test_values = [100, 105, 102, 98, 103, 101, 99, 104]
        ci = analyzer.compute_confidence_intervals(test_values)
        print(f"✓ Confidence interval: [{ci[0]:.2f}, {ci[1]:.2f}]")

        # Test normality test
        normality = analyzer.test_normality(test_values)
        print(f"✓ Normality test: p-value={normality['shapiro_pvalue']:.4f}, "
              f"is_normal={normality['is_normal']}")

        # Test learning curve analysis
        learning_analysis = analyzer.analyze_learning_curve(exp_id)
        print(f"✓ Learning analysis: final_performance={learning_analysis['final_performance']:.2f}, "
              f"trend={learning_analysis['trend']}")

        # Test create_summary_table
        table = analyzer.create_summary_table(exp_ids[:3], ['mean_reward', 'cache_hit_rate'])
        print(f"✓ Summary table: {table.shape[0]} rows × {table.shape[1]} columns")

        # Test identify_best_configuration
        best = analyzer.identify_best_configuration(exp_ids[:3], metric='mean_reward')
        print(f"✓ Best configuration: {best['best_name']} (mean={best['best_mean']:.2f})")

        print("\n✅ All ResultsAnalyzer tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ ResultsAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizer():
    """Test ResultsVisualizer functionality."""
    print("\n" + "="*80)
    print("TEST 2: ResultsVisualizer")
    print("="*80)

    # Create mock data
    test_dir = create_mock_results('results/test_visualizer', num_experiments=3)

    try:
        analyzer = ResultsAnalyzer(str(test_dir))
        visualizer = ResultsVisualizer()

        exp_ids = list(analyzer.experiments.keys())
        output_dir = Path('results/test_visualizer/plots')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test learning curves
        lc_path = output_dir / 'learning_curves.png'
        visualizer.plot_learning_curves(
            analyzer, exp_ids[:2],
            metric='reward',
            output_path=str(lc_path)
        )
        print(f"✓ Generated learning curves: {lc_path}")

        # Test training comparison
        tc_path = output_dir / 'training_comparison.png'
        visualizer.plot_training_comparison(
            analyzer, exp_ids[:2],
            metrics=['reward', 'loss'],
            output_path=str(tc_path)
        )
        print(f"✓ Generated training comparison: {tc_path}")

        # Test bar comparison
        bar_path = output_dir / 'bar_comparison.png'
        visualizer.plot_bar_comparison(
            analyzer, exp_ids[:3],
            metric='mean_reward',
            output_path=str(bar_path)
        )
        print(f"✓ Generated bar comparison: {bar_path}")

        # Test box plot
        box_path = output_dir / 'box_plot.png'
        visualizer.plot_box_comparison(
            analyzer, exp_ids[:3],
            metric='mean_reward',
            output_path=str(box_path)
        )
        print(f"✓ Generated box plot: {box_path}")

        # Test radar chart
        radar_path = output_dir / 'radar_chart.png'
        visualizer.plot_radar_chart(
            analyzer, exp_ids[:3],
            metrics=['mean_reward', 'cache_hit_rate'],
            output_path=str(radar_path)
        )
        print(f"✓ Generated radar chart: {radar_path}")

        # Test generate_all_thesis_figures
        figures = visualizer.generate_all_thesis_figures(
            analyzer, exp_ids[:3],
            str(output_dir / 'thesis'),
            metrics=['mean_reward', 'cache_hit_rate']
        )
        print(f"✓ Generated {len(figures)} thesis figures")

        print("\n✅ All ResultsVisualizer tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ ResultsVisualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generator():
    """Test ReportGenerator functionality."""
    print("\n" + "="*80)
    print("TEST 3: ReportGenerator")
    print("="*80)

    # Create mock data
    test_dir = create_mock_results('results/test_reports', num_experiments=3)

    try:
        analyzer = ResultsAnalyzer(str(test_dir))
        visualizer = ResultsVisualizer()
        generator = ReportGenerator(analyzer, visualizer)

        exp_ids = list(analyzer.experiments.keys())
        output_dir = Path('results/test_reports/reports')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test single experiment report (markdown)
        md_path = output_dir / 'experiment_report.md'
        generator.generate_experiment_report(
            exp_ids[0],
            str(md_path),
            format='markdown',
            include_plots=False
        )
        print(f"✓ Generated markdown report: {md_path}")

        # Test single experiment report (html)
        html_path = output_dir / 'experiment_report.html'
        generator.generate_experiment_report(
            exp_ids[0],
            str(html_path),
            format='html',
            include_plots=False
        )
        print(f"✓ Generated HTML report: {html_path}")

        # Test comparison report
        comp_path = output_dir / 'comparison_report.html'
        generator.generate_comparison_report(
            exp_ids[:3],
            str(comp_path),
            format='html',
            metrics=['mean_reward', 'cache_hit_rate'],
            include_plots=False
        )
        print(f"✓ Generated comparison report: {comp_path}")

        # Test thesis chapter
        thesis_path = output_dir / 'thesis_chapter.tex'
        generator.generate_thesis_chapter(
            str(test_dir),
            str(thesis_path)
        )
        print(f"✓ Generated thesis chapter: {thesis_path}")

        print("\n✅ All ReportGenerator tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ ReportGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test complete workflow integration."""
    print("\n" + "="*80)
    print("TEST 4: Integration Test")
    print("="*80)

    # Create mock data
    test_dir = create_mock_results('results/test_integration', num_experiments=3)

    try:
        # Complete workflow
        analyzer = ResultsAnalyzer(str(test_dir))
        visualizer = ResultsVisualizer()
        generator = ReportGenerator(analyzer, visualizer)

        exp_ids = list(analyzer.experiments.keys())

        # 1. Load and analyze
        print("\n1. Loading and analyzing experiments...")
        dqn_results = analyzer.load_experiment(exp_ids[0])
        lru_results = analyzer.load_experiment(exp_ids[1])
        print(f"   ✓ Loaded {len(dqn_results)} DQN results")
        print(f"   ✓ Loaded {len(lru_results)} LRU results")

        # 2. Statistical comparison
        print("\n2. Statistical comparison...")
        dqn_rewards = [r.final_eval_metrics['mean_reward'] for r in dqn_results]
        lru_rewards = [r.final_eval_metrics['mean_reward'] for r in lru_results]

        comparison = analyzer.compare_two_methods(dqn_rewards, lru_rewards, metric='mean_reward')
        print(f"   DQN vs LRU:")
        print(f"   - t-test p-value: {comparison['t_test_pvalue']:.4f}")
        print(f"   - Effect size (Cohen's d): {comparison['cohens_d']:.2f}")
        print(f"   - Significant: {comparison['significant']}")

        # 3. Multi-method comparison
        print("\n3. Multi-method comparison...")
        all_results = {}
        for exp_id in exp_ids[:3]:
            results = analyzer.load_experiment(exp_id)
            all_results[exp_id] = [r.final_eval_metrics['mean_reward'] for r in results]

        multi_comp = analyzer.compare_multiple_methods(all_results, metric='mean_reward')
        print(f"   ANOVA p-value: {multi_comp['anova_pvalue']:.4f}")
        print(f"   Rankings:")
        for rank, (name, mean) in enumerate(multi_comp['rankings'], 1):
            print(f"     {rank}. {name}: {mean:.2f}")

        # 4. Generate plots
        print("\n4. Generating plots...")
        output_dir = Path('results/test_integration/figures')
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer.plot_learning_curves(
            analyzer, exp_ids[:2],
            metric='reward',
            output_path=str(output_dir / 'learning.png')
        )
        print(f"   ✓ Learning curves")

        visualizer.plot_bar_comparison(
            analyzer, exp_ids[:3],
            metric='mean_reward',
            output_path=str(output_dir / 'comparison.png')
        )
        print(f"   ✓ Bar comparison")

        # 5. Generate reports
        print("\n5. Generating reports...")
        report_dir = Path('results/test_integration/reports')
        report_dir.mkdir(parents=True, exist_ok=True)

        generator.generate_comparison_report(
            exp_ids[:3],
            str(report_dir / 'comparison.html'),
            format='html',
            metrics=['mean_reward', 'cache_hit_rate'],
            include_plots=False
        )
        print(f"   ✓ Comparison report")

        print("\n✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("RESULTS ANALYSIS MODULE VALIDATION")
    print("="*80)

    results = {
        'Analyzer': test_analyzer(),
        'Visualizer': test_visualizer(),
        'ReportGenerator': test_report_generator(),
        'Integration': test_integration()
    }

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All tests passed! Module is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

