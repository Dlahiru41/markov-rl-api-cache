"""
Demo script for MarkovEvaluator and MarkovVisualizer.

Demonstrates comprehensive evaluation and visualization of Markov chain
prediction performance for thesis evaluation section.
"""

from src.markov import MarkovPredictor, MarkovEvaluator, MarkovVisualizer


def main():
    """Run the demonstration."""
    print("=" * 70)
    print("Markov Chain Evaluation - Comprehensive Demo")
    print("=" * 70)

    # 1. Setup and train models
    print("\n1. Training Markov chain models...")
    print("   " + "=" * 60)

    # Training data
    sequences = [
        ['login', 'profile', 'orders', 'product', 'cart'],
        ['login', 'profile', 'orders', 'product', 'reviews'],
        ['login', 'browse', 'product', 'cart', 'checkout'],
        ['login', 'browse', 'product', 'reviews', 'cart'],
        ['browse', 'search', 'product', 'cart', 'checkout'],
        ['browse', 'search', 'product', 'reviews', 'back'],
        ['login', 'profile', 'settings', 'security', 'logout'],
        ['login', 'profile', 'settings', 'privacy', 'logout'],
    ] * 3  # Repeat for more data

    # Train first-order model
    predictor_first = MarkovPredictor(order=1, history_size=10)
    predictor_first.fit(sequences)
    print("   [OK] First-order model trained")

    # Train second-order model
    predictor_second = MarkovPredictor(order=2, history_size=10)
    predictor_second.fit(sequences)
    print("   [OK] Second-order model trained")

    # 2. Basic evaluation
    print("\n2. Basic Evaluation Metrics")
    print("   " + "=" * 60)

    evaluator = MarkovEvaluator(predictor_first)
    results = evaluator.evaluate_accuracy(sequences, k_values=[1, 3, 5, 10])

    print(f"\n   First-Order Model:")
    print(f"      Top-1 Accuracy:  {results['top_1_accuracy']:.1%}")
    print(f"      Top-3 Accuracy:  {results['top_3_accuracy']:.1%}")
    print(f"      Top-5 Accuracy:  {results['top_5_accuracy']:.1%}")
    print(f"      Top-10 Accuracy: {results['top_10_accuracy']:.1%}")
    print(f"      MRR:             {results['mrr']:.3f}")
    print(f"      Coverage:        {results['coverage']:.1%}")
    print(f"      Perplexity:      {results['perplexity']:.2f}")

    # 3. Per-endpoint breakdown
    print("\n3. Per-Endpoint Breakdown")
    print("   " + "=" * 60)

    per_endpoint = evaluator.evaluate_per_endpoint(sequences, k_values=[1, 3])
    print(f"\n   Top 5 endpoints by sample count:")
    print(per_endpoint[['endpoint', 'sample_count', 'top_1_accuracy', 'top_3_accuracy', 'mrr']].head(5).to_string(index=False))

    # 4. Calibration
    print("\n4. Calibration Analysis")
    print("   " + "=" * 60)

    calibration = evaluator.evaluate_calibration(sequences, num_bins=5)
    print(f"\n   Calibration bins: {len(calibration['bin_centers'])}")
    if calibration['bin_centers']:
        print("   Predicted Prob  →  Actual Accuracy  (Samples)")
        for pred, actual, count in zip(
            calibration['predicted_probs'],
            calibration['actual_accuracy'],
            calibration['sample_counts']
        ):
            print(f"      {pred:.2f}         →  {actual:.2f}            ({int(count)})")

    # 5. Cross-validation
    print("\n5. Cross-Validation (3-fold)")
    print("   " + "=" * 60)

    print("\n   Running cross-validation...")
    cv_results = evaluator.cross_validate(sequences, k_folds=3, k_values=[1, 3, 5])

    print(f"\n   Results:")
    for metric in ['top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy', 'mrr']:
        if metric in cv_results:
            mean, std = cv_results[metric]
            print(f"      {metric:20s}: {mean:.3f} ± {std:.3f}")

    # 6. Model comparison
    print("\n6. Model Comparison")
    print("   " + "=" * 60)

    models = {
        'First-Order': predictor_first,
        'Second-Order': predictor_second
    }

    print("\n   Comparing models...")
    comparison = evaluator.compare_models(models, sequences, k_values=[1, 3, 5])

    print(f"\n   Results:")
    print(comparison[['model', 'top_1_accuracy', 'top_3_accuracy', 'top_5_accuracy', 'mrr']].to_string(index=False))

    # 7. Visualizations
    print("\n7. Generating Visualizations")
    print("   " + "=" * 60)

    try:
        # Transition heatmap
        print("\n   Creating transition heatmap...")
        MarkovVisualizer.plot_transition_heatmap(
            predictor_first,
            top_k=10,
            output_path='data/test/demo_heatmap.png'
        )

        # Accuracy by position
        print("   Creating accuracy by position plot...")
        MarkovVisualizer.plot_accuracy_by_position(
            sequences,
            predictor_first,
            max_position=4,
            output_path='data/test/demo_accuracy_by_position.png'
        )

        # Calibration curve
        print("   Creating calibration curve...")
        MarkovVisualizer.plot_calibration_curve(
            calibration,
            output_path='data/test/demo_calibration.png'
        )

        # Confidence distribution
        print("   Creating confidence distribution...")
        MarkovVisualizer.plot_prediction_confidence_distribution(
            sequences,
            predictor_first,
            output_path='data/test/demo_confidence_dist.png'
        )

        # Model comparison
        print("   Creating model comparison plot...")
        MarkovVisualizer.plot_model_comparison(
            comparison,
            metrics=['top_1_accuracy', 'top_3_accuracy', 'mrr'],
            output_path='data/test/demo_model_comparison.png'
        )

        print("\n   [OK] All visualizations saved to data/test/")

    except Exception as e:
        print(f"\n   ⚠ Visualizations skipped: {e}")
        print("   (matplotlib/seaborn may not be installed)")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Insights:")
    print("  • Comprehensive metrics for rigorous evaluation")
    print("  • Per-endpoint breakdown identifies prediction challenges")
    print("  • Calibration shows if probabilities are reliable")
    print("  • Cross-validation provides confidence intervals")
    print("  • Model comparison facilitates systematic selection")
    print("  • Visualizations support thesis evaluation section")
    print()


if __name__ == '__main__':
    main()

