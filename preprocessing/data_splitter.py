"""Data splitting utilities for creating train/test splits for ML experiments.

This module provides various strategies for splitting datasets to prevent data leakage
and ensure proper evaluation of machine learning models.
"""

from typing import List, Tuple, Optional
from datetime import datetime
import random
from collections import defaultdict

from preprocessing.models import Dataset


class DataSplitter:
    """Provides various strategies for splitting datasets into train/test sets.

    All splitting methods:
    - Never split a single session across train and test
    - Return Dataset objects
    - Respect the requested split ratio approximately
    - Are deterministic when given a random seed
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the data splitter.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def split_chronological(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset chronologically by session start time.

        Earlier sessions go to training, later sessions to testing.
        This prevents data leakage where we train on future data to predict the past.
        Important for time-series data like user behavior.

        Args:
            dataset: Dataset to split
            train_ratio: Fraction of sessions for training (0.0 to 1.0)

        Returns:
            Tuple of (train_dataset, test_dataset)

        Raises:
            ValueError: If train_ratio not in (0, 1)
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        # Sort sessions by start time
        sorted_sessions = sorted(dataset.sessions, key=lambda s: s.start_timestamp)

        # Calculate split point
        split_idx = int(len(sorted_sessions) * train_ratio)

        # Split sessions
        train_sessions = sorted_sessions[:split_idx]
        test_sessions = sorted_sessions[split_idx:]

        # Create datasets
        train_dataset = Dataset(
            name=f"{dataset.name}_train_chronological",
            sessions=train_sessions,
            metadata={
                **dataset.metadata,
                'split_type': 'chronological',
                'train_ratio': train_ratio,
                'parent_dataset': dataset.name
            }
        )

        test_dataset = Dataset(
            name=f"{dataset.name}_test_chronological",
            sessions=test_sessions,
            metadata={
                **dataset.metadata,
                'split_type': 'chronological',
                'train_ratio': train_ratio,
                'parent_dataset': dataset.name
            }
        )

        return train_dataset, test_dataset

    def split_by_users(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset by users.

        Some users appear only in training, others only in testing.
        Tests whether our model generalizes to completely new users.
        Randomly assigns users to train or test set.

        Args:
            dataset: Dataset to split
            train_ratio: Fraction of users for training (0.0 to 1.0)

        Returns:
            Tuple of (train_dataset, test_dataset)

        Raises:
            ValueError: If train_ratio not in (0, 1)
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        # Get unique users
        users = list(set(session.user_id for session in dataset.sessions))

        # Shuffle users
        if self.seed is not None:
            random.Random(self.seed).shuffle(users)
        else:
            random.shuffle(users)

        # Split users
        split_idx = int(len(users) * train_ratio)
        train_users = set(users[:split_idx])
        test_users = set(users[split_idx:])

        # Assign sessions to train or test based on user
        train_sessions = [s for s in dataset.sessions if s.user_id in train_users]
        test_sessions = [s for s in dataset.sessions if s.user_id in test_users]

        # Create datasets
        train_dataset = Dataset(
            name=f"{dataset.name}_train_users",
            sessions=train_sessions,
            metadata={
                **dataset.metadata,
                'split_type': 'user_based',
                'train_ratio': train_ratio,
                'train_users': len(train_users),
                'test_users': len(test_users),
                'parent_dataset': dataset.name
            }
        )

        test_dataset = Dataset(
            name=f"{dataset.name}_test_users",
            sessions=test_sessions,
            metadata={
                **dataset.metadata,
                'split_type': 'user_based',
                'train_ratio': train_ratio,
                'train_users': len(train_users),
                'test_users': len(test_users),
                'parent_dataset': dataset.name
            }
        )

        return train_dataset, test_dataset

    def split_stratified(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        stratify_by: str = 'user_type'
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset while maintaining distribution of a key attribute.

        Maintains the same distribution of the specified attribute in both train and test.
        Ensures both sets are representative of the overall population.

        Args:
            dataset: Dataset to split
            train_ratio: Fraction of sessions for training (0.0 to 1.0)
            stratify_by: Attribute to stratify by ('user_type' or other session attribute)

        Returns:
            Tuple of (train_dataset, test_dataset)

        Raises:
            ValueError: If train_ratio not in (0, 1) or invalid stratify_by attribute
        """
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        # Group sessions by stratification attribute
        stratified_groups = defaultdict(list)

        for session in dataset.sessions:
            if not hasattr(session, stratify_by):
                raise ValueError(f"Session does not have attribute '{stratify_by}'")

            key = getattr(session, stratify_by)
            stratified_groups[key].append(session)

        # Split each group proportionally
        train_sessions = []
        test_sessions = []

        for key, group_sessions in stratified_groups.items():
            # Shuffle sessions within group
            group_copy = group_sessions.copy()
            if self.seed is not None:
                random.Random(self.seed + hash(key)).shuffle(group_copy)
            else:
                random.shuffle(group_copy)

            # Split group
            split_idx = int(len(group_copy) * train_ratio)
            train_sessions.extend(group_copy[:split_idx])
            test_sessions.extend(group_copy[split_idx:])

        # Create datasets
        train_dataset = Dataset(
            name=f"{dataset.name}_train_stratified",
            sessions=train_sessions,
            metadata={
                **dataset.metadata,
                'split_type': 'stratified',
                'stratify_by': stratify_by,
                'train_ratio': train_ratio,
                'parent_dataset': dataset.name
            }
        )

        test_dataset = Dataset(
            name=f"{dataset.name}_test_stratified",
            sessions=test_sessions,
            metadata={
                **dataset.metadata,
                'split_type': 'stratified',
                'stratify_by': stratify_by,
                'train_ratio': train_ratio,
                'parent_dataset': dataset.name
            }
        )

        return train_dataset, test_dataset

    def k_fold_split(
        self,
        dataset: Dataset,
        k: int = 5,
        shuffle: bool = True
    ) -> List[Tuple[Dataset, Dataset]]:
        """Create K different train/test splits for cross-validation.

        Creates K folds where each fold uses a different portion as the test set.
        Enables more robust evaluation by testing on all data.

        Args:
            dataset: Dataset to split
            k: Number of folds (must be >= 2)
            shuffle: Whether to shuffle sessions before splitting

        Returns:
            List of K (train_dataset, test_dataset) tuples

        Raises:
            ValueError: If k < 2
        """
        if k < 2:
            raise ValueError(f"k must be at least 2, got {k}")

        # Get sessions
        sessions = dataset.sessions.copy()

        # Shuffle if requested
        if shuffle:
            if self.seed is not None:
                random.Random(self.seed).shuffle(sessions)
            else:
                random.shuffle(sessions)

        # Calculate fold size
        fold_size = len(sessions) // k

        # Create folds
        folds = []

        for fold_idx in range(k):
            # Determine test indices for this fold
            test_start = fold_idx * fold_size
            if fold_idx == k - 1:
                # Last fold gets any remaining sessions
                test_end = len(sessions)
            else:
                test_end = test_start + fold_size

            # Split sessions
            test_sessions = sessions[test_start:test_end]
            train_sessions = sessions[:test_start] + sessions[test_end:]

            # Create datasets
            train_dataset = Dataset(
                name=f"{dataset.name}_train_fold{fold_idx+1}",
                sessions=train_sessions,
                metadata={
                    **dataset.metadata,
                    'split_type': 'k_fold',
                    'fold': fold_idx + 1,
                    'total_folds': k,
                    'parent_dataset': dataset.name
                }
            )

            test_dataset = Dataset(
                name=f"{dataset.name}_test_fold{fold_idx+1}",
                sessions=test_sessions,
                metadata={
                    **dataset.metadata,
                    'split_type': 'k_fold',
                    'fold': fold_idx + 1,
                    'total_folds': k,
                    'parent_dataset': dataset.name
                }
            )

            folds.append((train_dataset, test_dataset))

        return folds

    def split_by_time_window(
        self,
        dataset: Dataset,
        test_start: datetime,
        test_end: Optional[datetime] = None
    ) -> Tuple[Dataset, Dataset]:
        """Split dataset by a specific time window.

        Sessions in the time window go to test, others to train.
        Useful for testing on a specific time period (e.g., a particular week).

        Args:
            dataset: Dataset to split
            test_start: Start of test window (inclusive)
            test_end: End of test window (exclusive), None for all sessions after test_start

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        train_sessions = []
        test_sessions = []

        for session in dataset.sessions:
            if test_end is None:
                # Everything after test_start goes to test
                if session.start_timestamp >= test_start:
                    test_sessions.append(session)
                else:
                    train_sessions.append(session)
            else:
                # Check if session is in the window
                if test_start <= session.start_timestamp < test_end:
                    test_sessions.append(session)
                else:
                    train_sessions.append(session)

        # Create datasets
        train_dataset = Dataset(
            name=f"{dataset.name}_train_timewindow",
            sessions=train_sessions,
            metadata={
                **dataset.metadata,
                'split_type': 'time_window',
                'test_start': test_start.isoformat(),
                'test_end': test_end.isoformat() if test_end else None,
                'parent_dataset': dataset.name
            }
        )

        test_dataset = Dataset(
            name=f"{dataset.name}_test_timewindow",
            sessions=test_sessions,
            metadata={
                **dataset.metadata,
                'split_type': 'time_window',
                'test_start': test_start.isoformat(),
                'test_end': test_end.isoformat() if test_end else None,
                'parent_dataset': dataset.name
            }
        )

        return train_dataset, test_dataset

    def verify_no_overlap(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset
    ) -> bool:
        """Verify that train and test datasets have no overlapping sessions.

        Args:
            train_dataset: Training dataset
            test_dataset: Testing dataset

        Returns:
            True if no overlap, False if there is overlap
        """
        train_session_ids = {s.session_id for s in train_dataset.sessions}
        test_session_ids = {s.session_id for s in test_dataset.sessions}

        return train_session_ids.isdisjoint(test_session_ids)

    def get_split_summary(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset
    ) -> dict:
        """Get summary statistics about a train/test split.

        Args:
            train_dataset: Training dataset
            test_dataset: Testing dataset

        Returns:
            Dictionary with split statistics
        """
        train_users = {s.user_id for s in train_dataset.sessions}
        test_users = {s.user_id for s in test_dataset.sessions}

        # User type distributions
        train_user_types = {}
        for session in train_dataset.sessions:
            train_user_types[session.user_type] = train_user_types.get(session.user_type, 0) + 1

        test_user_types = {}
        for session in test_dataset.sessions:
            test_user_types[session.user_type] = test_user_types.get(session.user_type, 0) + 1

        return {
            'train_sessions': len(train_dataset.sessions),
            'test_sessions': len(test_dataset.sessions),
            'train_calls': train_dataset.total_calls,
            'test_calls': test_dataset.total_calls,
            'train_users': len(train_users),
            'test_users': len(test_users),
            'overlapping_users': len(train_users & test_users),
            'train_ratio': len(train_dataset.sessions) / (len(train_dataset.sessions) + len(test_dataset.sessions)),
            'no_session_overlap': self.verify_no_overlap(train_dataset, test_dataset),
            'train_user_type_dist': train_user_types,
            'test_user_type_dist': test_user_types
        }

