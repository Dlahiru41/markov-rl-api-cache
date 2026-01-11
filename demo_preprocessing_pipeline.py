"""
Validation script to demonstrate the complete preprocessing pipeline workflow.

This script shows how to:
1. Generate synthetic data and preprocess it
2. Validate the outputs
3. Load and use the processed artifacts

Run: python demo_preprocessing_pipeline.py
"""

import sys
from pathlib import Path
import json
import pickle
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.models import Dataset


def main():
    print("=" * 70)
    print("  Preprocessing Pipeline Demo")
    print("=" * 70)
    print()

    # Step 1: Run the pipeline
    print("Step 1: Running preprocessing pipeline...")
    print("-" * 70)

    pipeline = PreprocessingPipeline(
        output_dir='data/demo_output',
        train_ratio=0.8,
        seed=42
    )

    results = pipeline.run(
        generate_synthetic=True,
        num_users=100,
        sessions_per_user=(5, 2)
    )

    print("\n✓ Pipeline completed successfully!")
    print()

    # Step 2: Verify outputs exist
    print("Step 2: Verifying output files...")
    print("-" * 70)

    output_dir = Path('data/demo_output')
    expected_files = [
        'train.parquet',
        'test.parquet',
        'sequences.json',
        'feature_engineer.pkl',
        'statistics.json',
        'report.md'
    ]

    for filename in expected_files:
        filepath = output_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename:<25} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {filename:<25} (MISSING!)")

    print()

    # Step 3: Load and inspect the data
    print("Step 3: Loading and inspecting processed data...")
    print("-" * 70)

    # Load statistics
    with open(output_dir / 'statistics.json', 'r') as f:
        stats = json.load(f)

    print(f"  Raw data:")
    print(f"    • Sessions: {stats['raw_sessions']:,}")
    print(f"    • API calls: {stats['raw_calls']:,}")
    print()
    print(f"  Train/Test split:")
    print(f"    • Train: {stats['train_sessions']:,} sessions, {stats['train_calls']:,} calls")
    print(f"    • Test: {stats['test_sessions']:,} sessions, {stats['test_calls']:,} calls")
    print()
    print(f"  Features:")
    print(f"    • Sequences: {stats['sequences_extracted']:,}")
    print(f"    • Feature dimension: {stats['feature_dimension']}")
    print()
    print(f"  Processing:")
    print(f"    • Duration: {stats['duration_seconds']:.2f} seconds")
    print(f"    • Validation errors: {stats['validation_errors']}")
    print()

    # Step 4: Load as pandas DataFrames
    print("Step 4: Loading as pandas DataFrames...")
    print("-" * 70)

    train_df = pd.read_parquet(output_dir / 'train.parquet')
    test_df = pd.read_parquet(output_dir / 'test.parquet')

    print(f"  Train DataFrame: {len(train_df)} rows, {len(train_df.columns)} columns")
    print(f"    Columns: {', '.join(train_df.columns[:5])}...")
    print()
    print(f"  Test DataFrame: {len(test_df)} rows, {len(test_df.columns)} columns")
    print()

    # Step 5: Load as Dataset objects
    print("Step 5: Loading as Dataset objects...")
    print("-" * 70)

    train_dataset = Dataset.load_from_parquet(output_dir / 'train.parquet')
    test_dataset = Dataset.load_from_parquet(output_dir / 'test.parquet')

    print(f"  Train Dataset:")
    print(f"    • Sessions: {len(train_dataset.sessions)}")
    print(f"    • Unique users: {train_dataset.num_unique_users}")
    print(f"    • Unique endpoints: {len(train_dataset.unique_endpoints)}")
    print()
    print(f"  Test Dataset:")
    print(f"    • Sessions: {len(test_dataset.sessions)}")
    print(f"    • Unique users: {test_dataset.num_unique_users}")
    print(f"    • Unique endpoints: {len(test_dataset.unique_endpoints)}")
    print()

    # Step 6: Load sequences
    print("Step 6: Loading sequences for Markov training...")
    print("-" * 70)

    with open(output_dir / 'sequences.json', 'r') as f:
        sequences = json.load(f)

    print(f"  Total sequences: {len(sequences)}")
    print(f"  Sample sequence (first 5 steps):")
    if sequences:
        print(f"    {' → '.join(sequences[0][:5])}...")
    print()

    # Step 7: Load feature engineer
    print("Step 7: Loading fitted feature engineer...")
    print("-" * 70)

    with open(output_dir / 'feature_engineer.pkl', 'rb') as f:
        feature_engineer = pickle.load(f)

    info = feature_engineer.get_feature_info()
    print(f"  Feature dimension: {info['feature_dim']}")
    print(f"  Unique endpoints: {info['num_endpoints']}")
    print(f"  Categories: {info['num_categories']}")
    print()

    # Transform a sample session
    if train_dataset.sessions:
        sample_session = train_dataset.sessions[0]
        features = feature_engineer.transform_session(sample_session)
        print(f"  Sample transformation:")
        print(f"    Session with {len(sample_session.calls)} calls")
        print(f"    → {len(features)} feature vectors")
        print(f"    → Each vector has {len(features[0])} dimensions")
    print()

    # Step 8: Display report summary
    print("Step 8: Report summary...")
    print("-" * 70)

    report_path = output_dir / 'report.md'
    with open(report_path, 'r', encoding='utf-8') as f:
        report_lines = f.readlines()

    print(f"  Report generated: {report_path}")
    print(f"  Report length: {len(report_lines)} lines")
    print()
    print("  First few lines:")
    for line in report_lines[:10]:
        print(f"    {line.rstrip()}")
    print()

    # Summary
    print("=" * 70)
    print("  ✓ Demo completed successfully!")
    print("=" * 70)
    print()
    print("Key takeaways:")
    print("  1. The pipeline orchestrates all preprocessing steps automatically")
    print("  2. Multiple output formats support different use cases")
    print("  3. Data can be loaded as DataFrames or Dataset objects")
    print("  4. The feature engineer is fitted and ready for inference")
    print("  5. Sequences are extracted and ready for Markov training")
    print()
    print("Next steps:")
    print("  • Train a Markov chain using sequences.json")
    print("  • Train an RL agent using the feature engineer")
    print("  • Evaluate on the test set")
    print()


if __name__ == '__main__':
    main()

