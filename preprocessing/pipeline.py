"""Unified preprocessing pipeline orchestrating all data preparation steps.

This pipeline combines all preprocessing components into a single, easy-to-use workflow.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Any, List
from datetime import datetime
import logging

from preprocessing.models import Dataset, APICall, Session
from preprocessing.validator import DataValidator
from preprocessing.data_splitter import DataSplitter
from preprocessing.sequence_builder import SequenceBuilder
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.synthetic_generator import SyntheticGenerator


class PreprocessingPipeline:
    """Orchestrates the complete preprocessing workflow.

    The pipeline workflow:
    1. Load raw data from file OR generate synthetic data
    2. Run validation and report quality issues
    3. Extract sessions from raw API calls (if needed)
    4. Build sequences for Markov chain training
    5. Fit FeatureEngineer on training data
    6. Split into train/test datasets (chronological)
    7. Save all processed artifacts
    8. Generate summary report

    Example:
        pipeline = PreprocessingPipeline(
            output_dir='data/processed',
            seed=42
        )

        results = pipeline.run(
            input_path='data/raw/logs.csv',
            # OR
            generate_synthetic=True,
            num_users=1000
        )

        print(f"Train data: {results['train_path']}")
        print(f"Test data: {results['test_path']}")
    """

    def __init__(
        self,
        output_dir: str = 'data/processed',
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the preprocessing pipeline.

        Args:
            output_dir: Directory to save processed artifacts
            train_ratio: Fraction of data for training (0.0 to 1.0)
            seed: Random seed for reproducibility
            config: Optional configuration dictionary
        """
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.seed = seed
        self.config = config or {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.validator = DataValidator()
        self.splitter = DataSplitter(seed=seed)
        self.sequence_builder = SequenceBuilder()
        self.feature_engineer = FeatureEngineer()

        # Track statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'duration_seconds': None,
            'input_source': None,
            'raw_sessions': 0,
            'raw_calls': 0,
            'train_sessions': 0,
            'train_calls': 0,
            'test_sessions': 0,
            'test_calls': 0,
            'validation_errors': 0,
            'validation_warnings': 0,
            'quality_metrics': {},
            'anomalies_detected': 0,
            'sequences_extracted': 0,
            'feature_dimension': 0
        }

    def run(
        self,
        input_path: Optional[str] = None,
        generate_synthetic: bool = False,
        num_users: int = 1000,
        sessions_per_user: tuple = (5, 2),
        validate_only: bool = False
    ) -> Dict[str, Any]:
        """Execute the complete preprocessing pipeline.

        Args:
            input_path: Path to input data file (CSV, JSON, or Parquet)
            generate_synthetic: Generate synthetic data instead of loading
            num_users: Number of users for synthetic generation
            sessions_per_user: (mean, std) for sessions per user
            validate_only: Only validate, don't process

        Returns:
            Dictionary of output file paths and statistics

        Raises:
            ValueError: If neither input_path nor generate_synthetic specified
            FileNotFoundError: If input file doesn't exist
        """
        self.stats['start_time'] = datetime.now()

        try:
            # Step 1: Load or generate data
            self.logger.info("="*70)
            self.logger.info("PREPROCESSING PIPELINE STARTED")
            self.logger.info("="*70)

            dataset = self._load_or_generate_data(
                input_path=input_path,
                generate_synthetic=generate_synthetic,
                num_users=num_users,
                sessions_per_user=sessions_per_user
            )

            self.stats['raw_sessions'] = len(dataset.sessions)
            self.stats['raw_calls'] = dataset.total_calls

            # Step 2: Validate data
            self.logger.info("\nStep 2: Validating data quality...")
            validation_result = self._validate_data(dataset)

            if not validation_result['valid']:
                self.logger.warning(
                    f"Validation found {len(validation_result['errors'])} errors"
                )
                if not self.config.get('allow_invalid_data', False):
                    raise ValueError(
                        f"Data validation failed with {len(validation_result['errors'])} errors. "
                        f"First error: {validation_result['errors'][0] if validation_result['errors'] else 'Unknown'}"
                    )

            # If validate_only, stop here
            if validate_only:
                self.logger.info("\n[OK] Validation complete (validate_only=True)")
                return self._generate_validation_report(dataset, validation_result)

            # Step 3: Split into train/test
            self.logger.info("\nStep 3: Splitting into train/test sets...")
            train_dataset, test_dataset = self._split_data(dataset)

            self.stats['train_sessions'] = len(train_dataset.sessions)
            self.stats['train_calls'] = train_dataset.total_calls
            self.stats['test_sessions'] = len(test_dataset.sessions)
            self.stats['test_calls'] = test_dataset.total_calls

            # Step 4: Build sequences for Markov training
            self.logger.info("\nStep 4: Building sequences for Markov chain...")
            sequences = self._build_sequences(train_dataset)

            self.stats['sequences_extracted'] = len(sequences)

            # Step 5: Fit feature engineer on training data
            self.logger.info("\nStep 5: Fitting feature engineer...")
            self._fit_feature_engineer(train_dataset)

            self.stats['feature_dimension'] = self.feature_engineer.get_feature_dim()

            # Finalize statistics before saving
            self.stats['end_time'] = datetime.now()
            self.stats['duration_seconds'] = (
                self.stats['end_time'] - self.stats['start_time']
            ).total_seconds()

            # Step 6: Save all artifacts
            self.logger.info("\nStep 6: Saving processed artifacts...")
            output_paths = self._save_artifacts(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                sequences=sequences,
                validation_result=validation_result
            )

            # Step 7: Generate report
            self.logger.info("\nStep 7: Generating preprocessing report...")
            report_path = self._generate_report(
                dataset=dataset,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                validation_result=validation_result
            )
            output_paths['report'] = str(report_path)


            self.logger.info("\n" + "="*70)
            self.logger.info("[OK] PREPROCESSING PIPELINE COMPLETE")
            self.logger.info("="*70)
            self.logger.info(f"\nProcessed {self.stats['raw_calls']} API calls")
            self.logger.info(f"Train: {self.stats['train_sessions']} sessions ({self.stats['train_calls']} calls)")
            self.logger.info(f"Test: {self.stats['test_sessions']} sessions ({self.stats['test_calls']} calls)")
            self.logger.info(f"Duration: {self.stats['duration_seconds']:.1f} seconds")
            self.logger.info(f"\nOutputs saved to: {self.output_dir}")

            return output_paths

        except Exception as e:
            self.logger.error(f"\n[FAIL] Pipeline failed: {e}")
            raise

    def _load_or_generate_data(
        self,
        input_path: Optional[str],
        generate_synthetic: bool,
        num_users: int,
        sessions_per_user: tuple
    ) -> Dataset:
        """Load data from file or generate synthetic data."""
        if generate_synthetic:
            self.logger.info("\nStep 1: Generating synthetic data...")
            self.logger.info(f"  Users: {num_users}")
            self.logger.info(f"  Sessions per user: {sessions_per_user}")

            generator = SyntheticGenerator(seed=self.seed)
            dataset = generator.generate_dataset(
                num_users=num_users,
                sessions_per_user=sessions_per_user,
                show_progress=True
            )

            self.stats['input_source'] = 'synthetic'
            self.logger.info(f"  [OK] Generated {len(dataset.sessions)} sessions")

        elif input_path:
            self.logger.info(f"\nStep 1: Loading data from {input_path}...")
            input_path = Path(input_path)

            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # Load based on file extension
            if input_path.suffix == '.parquet':
                dataset = Dataset.load_from_parquet(input_path)
            elif input_path.suffix == '.json':
                dataset = Dataset.from_dict(json.loads(input_path.read_text()))
            else:
                raise ValueError(f"Unsupported file format: {input_path.suffix}")

            self.stats['input_source'] = str(input_path)
            self.logger.info(f"  [OK] Loaded {len(dataset.sessions)} sessions")

        else:
            raise ValueError(
                "Must specify either input_path or generate_synthetic=True"
            )

        return dataset

    def _validate_data(self, dataset: Dataset) -> Dict[str, Any]:
        """Validate data quality."""
        # Run validation
        result = self.validator.validate_dataset(dataset)

        self.stats['validation_errors'] = len(result['errors'])
        self.stats['validation_warnings'] = len(result['warnings'])

        # Check quality metrics
        quality = self.validator.check_data_quality(dataset)
        self.stats['quality_metrics'] = quality

        # Detect anomalies
        anomalies = self.validator.detect_anomalies(dataset)
        self.stats['anomalies_detected'] = len(anomalies)

        self.logger.info(f"  Valid: {'YES' if result['valid'] else 'NO'}")
        self.logger.info(f"  Errors: {len(result['errors'])}")
        self.logger.info(f"  Warnings: {len(result['warnings'])}")
        self.logger.info(f"  Anomalies: {len(anomalies)}")
        self.logger.info(f"  Error rate: {quality['error_rate']:.1%}")

        if result['errors'] and len(result['errors']) <= 5:
            self.logger.warning("  First few errors:")
            for error in result['errors'][:5]:
                self.logger.warning(f"    • {error}")

        return result

    def _split_data(self, dataset: Dataset) -> tuple:
        """Split dataset into train and test."""
        train, test = self.splitter.split_chronological(
            dataset,
            train_ratio=self.train_ratio
        )

        # Verify no overlap
        if not self.splitter.verify_no_overlap(train, test):
            self.logger.warning("  ⚠ Session overlap detected between train and test!")
        else:
            self.logger.info("  [OK] No session overlap (proper split)")

        self.logger.info(f"  Train: {len(train.sessions)} sessions ({train.total_calls} calls)")
        self.logger.info(f"  Test: {len(test.sessions)} sessions ({test.total_calls} calls)")

        return train, test

    def _build_sequences(self, dataset: Dataset) -> List[List[str]]:
        """Build sequences for Markov chain training."""
        sequences = self.sequence_builder.build_sequences(dataset.sessions)

        self.logger.info(f"  [OK] Extracted {len(sequences)} sequences")

        # Show sample
        if sequences:
            sample = sequences[0][:5] if len(sequences[0]) > 5 else sequences[0]
            self.logger.info(f"  Sample: {' → '.join(sample)}...")

        return sequences

    def _fit_feature_engineer(self, dataset: Dataset) -> None:
        """Fit feature engineer on training data."""
        self.feature_engineer.fit(dataset.sessions)

        info = self.feature_engineer.get_feature_info()
        self.logger.info(f"  [OK] Fitted feature engineer")
        self.logger.info(f"  Feature dimension: {info['feature_dim']}")
        self.logger.info(f"  Unique endpoints: {info['num_endpoints']}")
        self.logger.info(f"  Categories: {info['num_categories']}")

    def _save_artifacts(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        sequences: List[List[str]],
        validation_result: Dict[str, Any]
    ) -> Dict[str, str]:
        """Save all processed artifacts."""
        output_paths = {}

        # Save train dataset
        train_path = self.output_dir / 'train.parquet'
        train_dataset.save_to_parquet(train_path)
        output_paths['train'] = str(train_path)
        self.logger.info(f"  [OK] Saved train data: {train_path}")

        # Save test dataset
        test_path = self.output_dir / 'test.parquet'
        test_dataset.save_to_parquet(test_path)
        output_paths['test'] = str(test_path)
        self.logger.info(f"  [OK] Saved test data: {test_path}")

        # Save sequences
        sequences_path = self.output_dir / 'sequences.json'
        sequences_path.write_text(json.dumps(sequences, indent=2))
        output_paths['sequences'] = str(sequences_path)
        self.logger.info(f"  [OK] Saved sequences: {sequences_path}")

        # Save fitted feature engineer
        fe_path = self.output_dir / 'feature_engineer.pkl'
        with open(fe_path, 'wb') as f:
            pickle.dump(self.feature_engineer, f)
        output_paths['feature_engineer'] = str(fe_path)
        self.logger.info(f"  [OK] Saved feature engineer: {fe_path}")

        # Save statistics
        stats_path = self.output_dir / 'statistics.json'
        stats_to_save = self._serialize_stats()
        stats_path.write_text(json.dumps(stats_to_save, indent=2))
        output_paths['statistics'] = str(stats_path)
        self.logger.info(f"  [OK] Saved statistics: {stats_path}")

        return output_paths

    def _serialize_stats(self) -> Dict[str, Any]:
        """Serialize statistics for JSON."""
        stats = self.stats.copy()

        # Convert datetime to string
        if stats['start_time']:
            stats['start_time'] = stats['start_time'].isoformat()
        if stats['end_time']:
            stats['end_time'] = stats['end_time'].isoformat()

        return stats

    def _generate_report(
        self,
        dataset: Dataset,
        train_dataset: Dataset,
        test_dataset: Dataset,
        validation_result: Dict[str, Any]
    ) -> Path:
        """Generate human-readable preprocessing report."""
        report_lines = [
            "# Preprocessing Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Pipeline Version:** 1.0",
            "",
            "---",
            "",
            "## Input Data",
            "",
            f"- **Source:** {self.stats['input_source']}",
            f"- **Total Sessions:** {self.stats['raw_sessions']:,}",
            f"- **Total API Calls:** {self.stats['raw_calls']:,}",
            f"- **Unique Users:** {dataset.num_unique_users}",
            f"- **Unique Endpoints:** {len(dataset.unique_endpoints)}",
            "",
            "## Data Validation",
            "",
            f"- **Valid:** {'[OK] YES' if validation_result['valid'] else '[FAIL] NO'}",
            f"- **Errors:** {self.stats['validation_errors']}",
            f"- **Warnings:** {self.stats['validation_warnings']}",
            f"- **Anomalies Detected:** {self.stats['anomalies_detected']}",
            "",
            "### Quality Metrics",
            "",
        ]

        for metric, value in self.stats['quality_metrics'].items():
            report_lines.append(f"- **{metric}:** {value:.1%}")

        report_lines.extend([
            "",
            "## Train/Test Split",
            "",
            f"- **Strategy:** Chronological (prevents time leakage)",
            f"- **Train Ratio:** {self.train_ratio:.1%}",
            "",
            "### Training Set",
            "",
            f"- **Sessions:** {self.stats['train_sessions']:,}",
            f"- **API Calls:** {self.stats['train_calls']:,}",
            "",
            "### Test Set",
            "",
            f"- **Sessions:** {self.stats['test_sessions']:,}",
            f"- **API Calls:** {self.stats['test_calls']:,}",
            "",
            "## Sequence Extraction",
            "",
            f"- **Sequences Extracted:** {self.stats['sequences_extracted']:,}",
            f"- **Purpose:** Markov chain training",
            "",
            "## Feature Engineering",
            "",
            f"- **Feature Dimension:** {self.stats['feature_dimension']}",
            f"- **Purpose:** RL state representation",
            "",
            "## Output Artifacts",
            "",
            f"**Location:** `{self.output_dir}`",
            "",
            "- `train.parquet` - Training dataset",
            "- `test.parquet` - Test dataset",
            "- `sequences.json` - Extracted sequences for Markov training",
            "- `feature_engineer.pkl` - Fitted feature engineer for inference",
            "- `statistics.json` - Detailed statistics in JSON format",
            "- `report.md` - This report",
            "",
            "## Processing Statistics",
            "",
        ])

        # Calculate duration if not already set
        if self.stats['duration_seconds'] is None and self.stats['start_time']:
            duration = (datetime.now() - self.stats['start_time']).total_seconds()
        else:
            duration = self.stats['duration_seconds'] or 0.0

        report_lines.extend([
            f"- **Duration:** {duration:.1f} seconds",
            f"- **Start Time:** {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S') if self.stats['start_time'] else 'N/A'}",
            f"- **End Time:** {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S') if self.stats['end_time'] else 'In Progress'}",
            "",
            "---",
            "",
            "**Pipeline Status:** [OK] Complete",
        ])

        report_path = self.output_dir / 'report.md'
        report_path.write_text('\n'.join(report_lines), encoding='utf-8')

        return report_path

    def _generate_validation_report(
        self,
        dataset: Dataset,
        validation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate validation-only report."""
        quality = self.validator.check_data_quality(dataset)
        anomalies = self.validator.detect_anomalies(dataset)

        return {
            'valid': validation_result['valid'],
            'errors': validation_result['errors'],
            'warnings': validation_result['warnings'],
            'quality_metrics': quality,
            'anomalies': anomalies,
            'total_sessions': len(dataset.sessions),
            'total_calls': dataset.total_calls
        }

