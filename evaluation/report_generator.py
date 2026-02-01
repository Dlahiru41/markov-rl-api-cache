"""
Automated report generation for experiment results.

Generates comprehensive reports in multiple formats (Markdown, HTML, LaTeX)
for thesis documentation and presentation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from jinja2 import Template

from .analyzer import ResultsAnalyzer, ResultsVisualizer


class ReportGenerator:
    """
    Generator for automated experiment reports.
    
    Creates comprehensive reports in various formats suitable for
    thesis documentation, presentations, and progress tracking.
    
    Features:
    - Single experiment reports
    - Multi-experiment comparison reports
    - Thesis chapter generation (LaTeX)
    - Statistical summaries
    - Figure integration
    
    Example:
        >>> analyzer = ResultsAnalyzer('results/experiments')
        >>> visualizer = ResultsVisualizer()
        >>> generator = ReportGenerator(analyzer, visualizer)
        >>> generator.generate_comparison_report(
        ...     ['dqn_baseline', 'lru_baseline'],
        ...     'reports/comparison.html'
        ... )
    """
    
    def __init__(self, analyzer: ResultsAnalyzer, visualizer: ResultsVisualizer):
        """
        Initialize report generator.
        
        Args:
            analyzer: ResultsAnalyzer instance
            visualizer: ResultsVisualizer instance
        """
        self.analyzer = analyzer
        self.visualizer = visualizer
    
    def generate_experiment_report(
        self,
        experiment_id: str,
        output_path: str,
        format: str = 'markdown',
        include_plots: bool = True
    ) -> str:
        """
        Generate complete report for one experiment.
        
        Args:
            experiment_id: Experiment identifier
            output_path: Path to save report
            format: Output format ('markdown', 'html', or 'latex')
            include_plots: Whether to generate and include plots
            
        Returns:
            Path to generated report
        """
        results = self.analyzer.load_experiment(experiment_id)
        if not results:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        # Generate summary
        summary = self.analyzer.summarize_experiment(experiment_id)
        
        # Analyze learning curve
        try:
            learning_analysis = self.analyzer.analyze_learning_curve(experiment_id)
        except:
            learning_analysis = None
        
        # Generate plots if requested
        plot_paths = {}
        if include_plots:
            output_dir = Path(output_path).parent / f"{experiment_id}_figures"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Learning curve
            try:
                lc_path = output_dir / 'learning_curve.png'
                self.visualizer.plot_learning_curves(
                    self.analyzer, [experiment_id],
                    output_path=str(lc_path)
                )
                plot_paths['learning_curve'] = str(lc_path)
            except Exception as e:
                print(f"Warning: Could not generate learning curve: {e}")
        
        # Build report content
        if format == 'markdown':
            content = self._generate_markdown_experiment_report(
                experiment_id, summary, learning_analysis, plot_paths
            )
        elif format == 'html':
            content = self._generate_html_experiment_report(
                experiment_id, summary, learning_analysis, plot_paths
            )
        elif format == 'latex':
            content = self._generate_latex_experiment_report(
                experiment_id, summary, learning_analysis, plot_paths
            )
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Generated {format} report: {output_path}")
        return output_path
    
    def generate_comparison_report(
        self,
        experiment_ids: List[str],
        output_path: str,
        format: str = 'html',
        metrics: List[str] = ['mean_reward', 'cache_hit_rate'],
        include_plots: bool = True
    ) -> str:
        """
        Generate comparison report across multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            output_path: Path to save report
            format: Output format ('markdown', 'html', or 'latex')
            metrics: Metrics to include in comparison
            include_plots: Whether to generate and include plots
            
        Returns:
            Path to generated report
        """
        # Collect experiment data
        exp_data = []
        for exp_id in experiment_ids:
            summary = self.analyzer.summarize_experiment(exp_id)
            exp_data.append(summary)
        
        # Statistical comparisons
        comparisons = {}
        for metric in metrics:
            # Collect values for this metric
            metric_values = {}
            for exp_id in experiment_ids:
                results = self.analyzer.load_experiment(exp_id)
                if results:
                    values = []
                    for r in results:
                        if r.status == 'completed' and metric in r.final_eval_metrics:
                            values.append(r.final_eval_metrics[metric])
                    if values:
                        metric_values[exp_id] = values
            
            if len(metric_values) > 1:
                # Multi-method comparison
                comparison = self.analyzer.compare_multiple_methods(metric_values, metric=metric)
                comparisons[metric] = comparison
        
        # Generate plots if requested
        plot_paths = {}
        if include_plots:
            output_dir = Path(output_path).parent / "comparison_figures"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Bar comparisons
            for metric in metrics:
                try:
                    bar_path = output_dir / f'bar_{metric}.png'
                    self.visualizer.plot_bar_comparison(
                        self.analyzer, experiment_ids, metric,
                        output_path=str(bar_path)
                    )
                    plot_paths[f'bar_{metric}'] = str(bar_path)
                except Exception as e:
                    print(f"Warning: Could not generate bar chart for {metric}: {e}")
            
            # Learning curves
            for metric in metrics:
                try:
                    lc_path = output_dir / f'learning_{metric}.png'
                    self.visualizer.plot_learning_curves(
                        self.analyzer, experiment_ids, metric=metric,
                        output_path=str(lc_path)
                    )
                    plot_paths[f'learning_{metric}'] = str(lc_path)
                except Exception as e:
                    print(f"Warning: Could not generate learning curve for {metric}: {e}")
        
        # Build report
        if format == 'markdown':
            content = self._generate_markdown_comparison_report(
                experiment_ids, exp_data, comparisons, metrics, plot_paths
            )
        elif format == 'html':
            content = self._generate_html_comparison_report(
                experiment_ids, exp_data, comparisons, metrics, plot_paths
            )
        elif format == 'latex':
            content = self._generate_latex_comparison_report(
                experiment_ids, exp_data, comparisons, metrics, plot_paths
            )
        else:
            raise ValueError(f"Unknown format: {format}")
        
        # Write report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Generated comparison report: {output_path}")
        return output_path
    
    def generate_thesis_chapter(
        self,
        results_dir: str,
        output_path: str,
        chapter_title: str = "Experimental Results",
        include_all_experiments: bool = False
    ) -> str:
        """
        Generate draft LaTeX chapter for thesis.
        
        Creates a complete results chapter with:
        - Experimental setup description
        - Baseline comparisons
        - Ablation studies
        - Statistical analysis
        - Figures and tables
        
        Args:
            results_dir: Directory with all experiment results
            output_path: Path to save LaTeX file
            chapter_title: Title for the chapter
            include_all_experiments: Whether to include all experiments or filter
            
        Returns:
            Path to generated LaTeX file
        """
        # Load all experiments
        experiments = self.analyzer.experiments
        
        # Organize experiments by tag
        by_tag = {}
        for exp_id, results in experiments.items():
            if results:
                tags = results[0].config.tags
                for tag in tags:
                    if tag not in by_tag:
                        by_tag[tag] = []
                    by_tag[tag].append(exp_id)
        
        # Generate figures
        figures_dir = Path(output_path).parent / 'thesis_figures'
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Build LaTeX content
        latex = self._generate_thesis_chapter_latex(
            chapter_title, experiments, by_tag, str(figures_dir)
        )
        
        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex)
        
        print(f"✓ Generated thesis chapter: {output_path}")
        return output_path
    
    # ========== Markdown Generation ==========
    
    def _generate_markdown_experiment_report(
        self,
        experiment_id: str,
        summary: Dict,
        learning_analysis: Optional[Dict],
        plot_paths: Dict[str, str]
    ) -> str:
        """Generate Markdown report for single experiment."""
        lines = []
        lines.append(f"# Experiment Report: {summary['name']}")
        lines.append(f"\n**Experiment ID:** `{experiment_id}`")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n---\n")
        
        # Configuration
        lines.append("## Configuration\n")
        lines.append(f"- **Number of Seeds:** {summary['num_seeds']}")
        lines.append(f"- **Completed Runs:** {summary['num_completed']}")
        lines.append("")
        
        # Results
        lines.append("## Results Summary\n")
        lines.append("| Metric | Mean | Std | Min | Max | Median | 95% CI |")
        lines.append("|--------|------|-----|-----|-----|--------|--------|")
        
        for metric, stats in summary['metrics'].items():
            ci_lower, ci_upper = stats['ci_95']
            lines.append(
                f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['median']:.4f} | "
                f"[{ci_lower:.4f}, {ci_upper:.4f}] |"
            )
        
        lines.append("")
        
        # Learning analysis
        if learning_analysis:
            lines.append("## Learning Analysis\n")
            lines.append(f"- **Total Episodes:** {learning_analysis['total_episodes']}")
            if learning_analysis['convergence_episode']:
                lines.append(f"- **Convergence Episode:** {learning_analysis['convergence_episode']}")
            lines.append(f"- **Final Performance:** {learning_analysis['final_performance']:.4f} ± {learning_analysis['final_std']:.4f}")
            lines.append(f"- **Stability (CV):** {learning_analysis['stability_cv']:.4f}")
            lines.append(f"- **Trend:** {learning_analysis['trend']} (slope={learning_analysis['trend_slope']:.4f})")
            lines.append("")
        
        # Plots
        if plot_paths:
            lines.append("## Visualizations\n")
            for plot_name, plot_path in plot_paths.items():
                lines.append(f"### {plot_name.replace('_', ' ').title()}\n")
                lines.append(f"![{plot_name}]({plot_path})\n")
        
        return "\n".join(lines)
    
    def _generate_markdown_comparison_report(
        self,
        experiment_ids: List[str],
        exp_data: List[Dict],
        comparisons: Dict,
        metrics: List[str],
        plot_paths: Dict[str, str]
    ) -> str:
        """Generate Markdown comparison report."""
        lines = []
        lines.append("# Experiment Comparison Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\n**Experiments Compared:** {len(experiment_ids)}")
        lines.append("\n---\n")
        
        # Summary table
        lines.append("## Summary Table\n")
        lines.append("| Experiment | " + " | ".join([f"{m} (mean±std)" for m in metrics]) + " |")
        lines.append("|------------|" + "|".join(["---"] * len(metrics)) + "|")
        
        for exp_summary in exp_data:
            row = [exp_summary['name']]
            for metric in metrics:
                if metric in exp_summary['metrics']:
                    stats = exp_summary['metrics'][metric]
                    row.append(f"{stats['mean']:.3f}±{stats['std']:.3f}")
                else:
                    row.append("N/A")
            lines.append("| " + " | ".join(row) + " |")
        
        lines.append("")
        
        # Statistical comparisons
        lines.append("## Statistical Analysis\n")
        for metric, comparison in comparisons.items():
            lines.append(f"### {metric.replace('_', ' ').title()}\n")
            lines.append(f"- **ANOVA p-value:** {comparison['anova_pvalue']:.4f}")
            lines.append(f"- **Significant:** {'Yes' if comparison['significant'] else 'No'}")
            lines.append(f"\n**Rankings:**\n")
            for rank, (name, mean) in enumerate(comparison['rankings'], 1):
                lines.append(f"{rank}. {name}: {mean:.4f}")
            lines.append("")
        
        # Plots
        if plot_paths:
            lines.append("## Visualizations\n")
            for plot_name, plot_path in plot_paths.items():
                lines.append(f"### {plot_name.replace('_', ' ').title()}\n")
                lines.append(f"![{plot_name}]({plot_path})\n")
        
        return "\n".join(lines)
    
    # ========== HTML Generation ==========
    
    def _generate_html_experiment_report(
        self,
        experiment_id: str,
        summary: Dict,
        learning_analysis: Optional[Dict],
        plot_paths: Dict[str, str]
    ) -> str:
        """Generate HTML report for single experiment."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Experiment Report: {{ name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .metric-value { font-weight: bold; color: #2196F3; }
        .plot { margin: 20px 0; text-align: center; }
        .plot img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        .info-box { background: #e3f2fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Experiment Report: {{ name }}</h1>
        <p class="timestamp">Experiment ID: <code>{{ exp_id }}</code></p>
        <p class="timestamp">Generated: {{ timestamp }}</p>
        
        <div class="info-box">
            <strong>Configuration:</strong>
            <ul>
                <li>Number of Seeds: {{ num_seeds }}</li>
                <li>Completed Runs: {{ num_completed }}</li>
            </ul>
        </div>
        
        <h2>Results Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Median</th>
                    <th>95% CI</th>
                </tr>
            </thead>
            <tbody>
                {% for metric, stats in metrics.items() %}
                <tr>
                    <td>{{ metric }}</td>
                    <td class="metric-value">{{ "%.4f"|format(stats.mean) }}</td>
                    <td>{{ "%.4f"|format(stats.std) }}</td>
                    <td>{{ "%.4f"|format(stats.min) }}</td>
                    <td>{{ "%.4f"|format(stats.max) }}</td>
                    <td>{{ "%.4f"|format(stats.median) }}</td>
                    <td>[{{ "%.4f"|format(stats.ci_95[0]) }}, {{ "%.4f"|format(stats.ci_95[1]) }}]</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        {% if learning_analysis %}
        <h2>Learning Analysis</h2>
        <div class="info-box">
            <ul>
                <li>Total Episodes: {{ learning_analysis.total_episodes }}</li>
                {% if learning_analysis.convergence_episode %}
                <li>Convergence Episode: {{ learning_analysis.convergence_episode }}</li>
                {% endif %}
                <li>Final Performance: {{ "%.4f"|format(learning_analysis.final_performance) }} ± {{ "%.4f"|format(learning_analysis.final_std) }}</li>
                <li>Stability (CV): {{ "%.4f"|format(learning_analysis.stability_cv) }}</li>
                <li>Trend: {{ learning_analysis.trend }} (slope={{ "%.4f"|format(learning_analysis.trend_slope) }})</li>
            </ul>
        </div>
        {% endif %}
        
        {% if plot_paths %}
        <h2>Visualizations</h2>
        {% for plot_name, plot_path in plot_paths.items() %}
        <div class="plot">
            <h3>{{ plot_name.replace('_', ' ').title() }}</h3>
            <img src="{{ plot_path }}" alt="{{ plot_name }}">
        </div>
        {% endfor %}
        {% endif %}
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(
            exp_id=experiment_id,
            name=summary['name'],
            num_seeds=summary['num_seeds'],
            num_completed=summary['num_completed'],
            metrics=summary['metrics'],
            learning_analysis=learning_analysis,
            plot_paths=plot_paths,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def _generate_html_comparison_report(
        self,
        experiment_ids: List[str],
        exp_data: List[Dict],
        comparisons: Dict,
        metrics: List[str],
        plot_paths: Dict[str, str]
    ) -> str:
        """Generate HTML comparison report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Experiment Comparison Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: auto; background: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        h3 { color: #666; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .best { background-color: #c8e6c9 !important; font-weight: bold; }
        .plot { margin: 30px 0; text-align: center; }
        .plot img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        .stats-box { background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }
        .significant { color: #f44336; font-weight: bold; }
        .not-significant { color: #9e9e9e; }
        ol { line-height: 1.8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Experiment Comparison Report</h1>
        <p>Generated: {{ timestamp }}</p>
        <p>Experiments Compared: {{ num_experiments }}</p>
        
        <h2>Summary Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Experiment</th>
                    {% for metric in metrics %}
                    <th>{{ metric.replace('_', ' ').title() }} (mean±std)</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for exp_summary in exp_data %}
                <tr>
                    <td>{{ exp_summary.name }}</td>
                    {% for metric in metrics %}
                    {% if metric in exp_summary.metrics %}
                    <td>{{ "%.3f"|format(exp_summary.metrics[metric].mean) }}±{{ "%.3f"|format(exp_summary.metrics[metric].std) }}</td>
                    {% else %}
                    <td>N/A</td>
                    {% endif %}
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <h2>Statistical Analysis</h2>
        {% for metric, comparison in comparisons.items() %}
        <div class="stats-box">
            <h3>{{ metric.replace('_', ' ').title() }}</h3>
            <p>ANOVA p-value: <strong>{{ "%.4f"|format(comparison.anova_pvalue) }}</strong></p>
            <p>Significant: <span class="{% if comparison.significant %}significant{% else %}not-significant{% endif %}">
                {{ 'Yes' if comparison.significant else 'No' }}
            </span></p>
            <p><strong>Rankings:</strong></p>
            <ol>
                {% for name, mean in comparison.rankings %}
                <li>{{ name }}: {{ "%.4f"|format(mean) }}</li>
                {% endfor %}
            </ol>
        </div>
        {% endfor %}
        
        {% if plot_paths %}
        <h2>Visualizations</h2>
        {% for plot_name, plot_path in plot_paths.items() %}
        <div class="plot">
            <h3>{{ plot_name.replace('_', ' ').title() }}</h3>
            <img src="{{ plot_path }}" alt="{{ plot_name }}">
        </div>
        {% endfor %}
        {% endif %}
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        return template.render(
            num_experiments=len(experiment_ids),
            exp_data=exp_data,
            comparisons=comparisons,
            metrics=metrics,
            plot_paths=plot_paths,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
    
    # ========== LaTeX Generation ==========
    
    def _generate_latex_experiment_report(
        self,
        experiment_id: str,
        summary: Dict,
        learning_analysis: Optional[Dict],
        plot_paths: Dict[str, str]
    ) -> str:
        """Generate LaTeX report for single experiment."""
        lines = []
        lines.append(r"\documentclass{article}")
        lines.append(r"\usepackage{booktabs}")
        lines.append(r"\usepackage{graphicx}")
        lines.append(r"\usepackage{float}")
        lines.append(r"\begin{document}")
        lines.append("")
        lines.append(rf"\section*{{Experiment Report: {summary['name']}}}")
        lines.append("")
        lines.append(rf"\textbf{{Experiment ID:}} \texttt{{{experiment_id}}}")
        lines.append(rf"\textbf{{Generated:}} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Results table
        lines.append(r"\subsection*{Results Summary}")
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\toprule")
        lines.append(r"Metric & Mean & Std & Min & Max & Median & 95\% CI \\")
        lines.append(r"\midrule")
        
        for metric, stats in summary['metrics'].items():
            ci_lower, ci_upper = stats['ci_95']
            metric_name = metric.replace('_', r'\_')
            lines.append(
                f"{metric_name} & {stats['mean']:.4f} & {stats['std']:.4f} & "
                f"{stats['min']:.4f} & {stats['max']:.4f} & {stats['median']:.4f} & "
                f"[{ci_lower:.4f}, {ci_upper:.4f}] \\\\"
            )
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Experiment results summary}")
        lines.append(r"\end{table}")
        lines.append("")
        
        # Learning analysis
        if learning_analysis:
            lines.append(r"\subsection*{Learning Analysis}")
            lines.append(r"\begin{itemize}")
            lines.append(rf"\item Total Episodes: {learning_analysis['total_episodes']}")
            if learning_analysis['convergence_episode']:
                lines.append(rf"\item Convergence Episode: {learning_analysis['convergence_episode']}")
            lines.append(rf"\item Final Performance: {learning_analysis['final_performance']:.4f} $\pm$ {learning_analysis['final_std']:.4f}")
            lines.append(rf"\item Trend: {learning_analysis['trend']}")
            lines.append(r"\end{itemize}")
            lines.append("")
        
        lines.append(r"\end{document}")
        return "\n".join(lines)
    
    def _generate_latex_comparison_report(
        self,
        experiment_ids: List[str],
        exp_data: List[Dict],
        comparisons: Dict,
        metrics: List[str],
        plot_paths: Dict[str, str]
    ) -> str:
        """Generate LaTeX comparison report."""
        lines = []
        lines.append(r"\documentclass{article}")
        lines.append(r"\usepackage{booktabs}")
        lines.append(r"\usepackage{graphicx}")
        lines.append(r"\usepackage{float}")
        lines.append(r"\begin{document}")
        lines.append("")
        lines.append(r"\section*{Experiment Comparison Report}")
        lines.append("")
        
        # Summary table
        lines.append(r"\subsection*{Summary Table}")
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{l" + "c" * len(metrics) + "}")
        lines.append(r"\toprule")
        header = "Experiment & " + " & ".join([m.replace('_', r'\_') for m in metrics]) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")
        
        for exp_summary in exp_data:
            row = [exp_summary['name'].replace('_', r'\_')]
            for metric in metrics:
                if metric in exp_summary['metrics']:
                    stats = exp_summary['metrics'][metric]
                    row.append(f"${stats['mean']:.3f} \\pm {stats['std']:.3f}$")
                else:
                    row.append("N/A")
            lines.append(" & ".join(row) + r" \\")
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Comparison of experiment results}")
        lines.append(r"\end{table}")
        lines.append("")
        
        lines.append(r"\end{document}")
        return "\n".join(lines)
    
    def _generate_thesis_chapter_latex(
        self,
        chapter_title: str,
        experiments: Dict,
        by_tag: Dict,
        figures_dir: str
    ) -> str:
        """Generate complete thesis chapter in LaTeX."""
        lines = []
        lines.append(r"\chapter{" + chapter_title + "}")
        lines.append("")
        lines.append(r"This chapter presents the experimental results of our Markov-RL based caching system.")
        lines.append("")
        
        # Experimental setup
        lines.append(r"\section{Experimental Setup}")
        lines.append(r"We conducted a comprehensive evaluation consisting of:")
        lines.append(r"\begin{itemize}")
        lines.append(rf"\item Total experiments: {len(experiments)}")
        lines.append(rf"\item Experiment categories: {len(by_tag)}")
        lines.append(r"\item Multiple random seeds for statistical significance")
        lines.append(r"\end{itemize}")
        lines.append("")
        
        # Baseline comparisons
        if 'baseline' in by_tag:
            lines.append(r"\section{Baseline Comparisons}")
            lines.append(r"We compared our RL-based approach against standard caching baselines:")
            lines.append(r"\begin{itemize}")
            lines.append(r"\item LRU (Least Recently Used)")
            lines.append(r"\item LFU (Least Frequently Used)")
            lines.append(r"\item Random replacement")
            lines.append(r"\item Static Markov policy")
            lines.append(r"\end{itemize}")
            lines.append("")
            lines.append(r"Figure~\ref{fig:baseline_comparison} shows the comparison results.")
            lines.append("")
        
        # Hyperparameter studies
        if 'hyperparameter_sweep' in by_tag:
            lines.append(r"\section{Hyperparameter Sensitivity}")
            lines.append(r"We performed systematic hyperparameter sweeps to identify optimal configurations.")
            lines.append("")
        
        # Ablation studies
        if 'ablation' in by_tag:
            lines.append(r"\section{Ablation Study}")
            lines.append(r"To understand the contribution of each component, we conducted ablation studies.")
            lines.append("")
        
        # Statistical analysis
        lines.append(r"\section{Statistical Analysis}")
        lines.append(r"All comparisons were tested for statistical significance using t-tests and ANOVA.")
        lines.append(r"We report 95\% confidence intervals throughout.")
        lines.append("")
        
        # Discussion
        lines.append(r"\section{Discussion}")
        lines.append(r"The results demonstrate that...")
        lines.append("")
        
        return "\n".join(lines)

