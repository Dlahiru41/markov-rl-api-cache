"""
Comprehensive unit tests for TransitionMatrix class.

Tests cover all functionality including:
- Basic operations (increment, get_count, get_probability)
- Top-k queries with heap optimization
- Matrix operations (merge, statistics)
- Serialization (to_dict, from_dict, save, load)
- Laplace smoothing
- Edge cases and error handling
- DataFrame conversion
"""

import pytest
import tempfile
from pathlib import Path

from src.markov.transition_matrix import TransitionMatrix


class TestBasicOperations:
    """Test basic increment, count, and probability operations."""

    def test_increment_and_get_count(self):
        """Test incrementing transition counts."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 5)

        assert tm.get_count("A", "B") == 5
        assert tm.get_count("A", "C") == 0  # Unseen transition

    def test_increment_default_count(self):
        """Test increment with default count of 1."""
        tm = TransitionMatrix()
        tm.increment("A", "B")

        assert tm.get_count("A", "B") == 1

    def test_increment_multiple_times(self):
        """Test multiple increments accumulate."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 3)
        tm.increment("A", "B", 2)

        assert tm.get_count("A", "B") == 5

    def test_get_probability_basic(self):
        """Test basic probability calculation."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 8)
        tm.increment("A", "C", 2)

        assert tm.get_probability("A", "B") == 0.8
        assert tm.get_probability("A", "C") == 0.2

    def test_get_probability_unseen_state(self):
        """Test probability for unseen state returns 0."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 10)

        assert tm.get_probability("Z", "X") == 0.0

    def test_get_probability_unseen_transition(self):
        """Test probability for unseen transition from known state."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 10)

        assert tm.get_probability("A", "C") == 0.0

    def test_get_row(self):
        """Test getting all transitions from a state."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 8)
        tm.increment("A", "C", 2)

        row = tm.get_row("A")

        assert len(row) == 2
        assert row["B"] == 0.8
        assert row["C"] == 0.2
        assert abs(sum(row.values()) - 1.0) < 1e-10  # Sum to 1

    def test_get_row_empty(self):
        """Test getting row for unseen state returns empty dict."""
        tm = TransitionMatrix()

        assert tm.get_row("Z") == {}


class TestTopK:
    """Test top-k query functionality."""

    def test_get_top_k_basic(self):
        """Test basic top-k query."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 10)
        tm.increment("A", "C", 5)
        tm.increment("A", "D", 3)

        top = tm.get_top_k("A", k=2)

        assert len(top) == 2
        assert top[0][0] == "B"  # Most likely
        assert top[1][0] == "C"  # Second most likely
        assert top[0][1] > top[1][1]  # Probabilities in descending order

    def test_get_top_k_all(self):
        """Test top-k when k >= number of transitions."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 10)
        tm.increment("A", "C", 5)

        top = tm.get_top_k("A", k=10)

        assert len(top) == 2  # Only 2 transitions exist
        assert top[0][0] == "B"

    def test_get_top_k_empty(self):
        """Test top-k for unseen state."""
        tm = TransitionMatrix()

        assert tm.get_top_k("Z", k=5) == []

    def test_get_top_k_zero(self):
        """Test top-k with k=0."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 10)

        assert tm.get_top_k("A", k=0) == []

    def test_get_top_k_ordering(self):
        """Test that top-k returns items in descending probability order."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 100)
        tm.increment("A", "C", 50)
        tm.increment("A", "D", 30)
        tm.increment("A", "E", 10)

        top = tm.get_top_k("A", k=3)

        # Check descending order
        assert top[0][1] > top[1][1] > top[2][1]
        assert top[0][0] == "B"
        assert top[1][0] == "C"
        assert top[2][0] == "D"


class TestSmoothing:
    """Test Laplace smoothing functionality."""

    def test_smoothing_initialization(self):
        """Test that smoothing parameter is stored."""
        tm = TransitionMatrix(smoothing=0.1)

        assert tm.smoothing == 0.1

    def test_no_smoothing_zero_probability(self):
        """Test that unseen transitions have 0 probability without smoothing."""
        tm = TransitionMatrix(smoothing=0.0)
        tm.increment("A", "B", 10)

        assert tm.get_probability("A", "C") == 0.0

    def test_smoothing_nonzero_probability(self):
        """Test that unseen transitions have non-zero probability with smoothing."""
        tm = TransitionMatrix(smoothing=0.1)
        tm.increment("A", "B", 10)
        tm.increment("B", "C", 5)  # Add another state to increase vocab

        prob = tm.get_probability("A", "C")

        assert prob > 0.0
        assert prob < tm.get_probability("A", "B")  # Should be less than seen transition

    def test_smoothing_negative_raises(self):
        """Test that negative smoothing raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            TransitionMatrix(smoothing=-0.1)


class TestMatrixProperties:
    """Test matrix-level properties and statistics."""

    def test_num_states(self):
        """Test counting unique states."""
        tm = TransitionMatrix()
        tm.increment("A", "B")
        tm.increment("B", "C")
        tm.increment("C", "A")

        # States: A, B, C
        assert tm.num_states == 3

    def test_num_states_empty(self):
        """Test num_states for empty matrix."""
        tm = TransitionMatrix()

        assert tm.num_states == 0

    def test_num_transitions(self):
        """Test counting non-zero transitions."""
        tm = TransitionMatrix()
        tm.increment("A", "B")
        tm.increment("A", "C")
        tm.increment("B", "C")

        assert tm.num_transitions == 3

    def test_sparsity(self):
        """Test sparsity calculation."""
        tm = TransitionMatrix()
        tm.increment("A", "B")
        tm.increment("A", "C")
        # 3 states (A, B, C) -> 3x3 = 9 possible transitions
        # 2 non-zero transitions -> 7 zeros
        # Sparsity = 7/9 ≈ 0.778

        assert abs(tm.sparsity - 7/9) < 1e-10

    def test_sparsity_empty(self):
        """Test sparsity for empty matrix."""
        tm = TransitionMatrix()

        assert tm.sparsity == 0.0

    def test_get_statistics(self):
        """Test statistics generation."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 100)
        tm.increment("A", "C", 50)
        tm.increment("B", "C", 30)

        stats = tm.get_statistics()

        assert stats["num_states"] == 3
        assert stats["num_transitions"] == 3
        assert "sparsity" in stats
        assert "avg_transitions_per_state" in stats
        assert "most_common_transitions" in stats

        # Check most common transition
        top_trans = stats["most_common_transitions"][0]
        assert top_trans["from"] == "A"
        assert top_trans["to"] == "B"
        assert top_trans["count"] == 100


class TestMerge:
    """Test matrix merging functionality."""

    def test_merge_basic(self):
        """Test merging two matrices."""
        tm1 = TransitionMatrix()
        tm1.increment("A", "B", 10)

        tm2 = TransitionMatrix()
        tm2.increment("A", "B", 5)
        tm2.increment("A", "C", 3)

        merged = tm1.merge(tm2)

        assert merged.get_count("A", "B") == 15
        assert merged.get_count("A", "C") == 3

    def test_merge_preserves_smoothing(self):
        """Test that merge preserves smoothing from first matrix."""
        tm1 = TransitionMatrix(smoothing=0.1)
        tm1.increment("A", "B", 10)

        tm2 = TransitionMatrix(smoothing=0.5)
        tm2.increment("C", "D", 5)

        merged = tm1.merge(tm2)

        assert merged.smoothing == 0.1

    def test_merge_empty(self):
        """Test merging with empty matrix."""
        tm1 = TransitionMatrix()
        tm1.increment("A", "B", 10)

        tm2 = TransitionMatrix()

        merged = tm1.merge(tm2)

        assert merged.get_count("A", "B") == 10


class TestSerialization:
    """Test serialization and deserialization."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        tm = TransitionMatrix(smoothing=0.1)
        tm.increment("A", "B", 5)
        tm.increment("A", "C", 3)

        data = tm.to_dict()

        assert data["smoothing"] == 0.1
        assert "transitions" in data
        assert "total_from" in data
        assert data["transitions"]["A"]["B"] == 5
        assert data["transitions"]["A"]["C"] == 3

    def test_from_dict(self):
        """Test reconstructing from dictionary."""
        data = {
            "smoothing": 0.1,
            "transitions": {
                "A": {"B": 5, "C": 3}
            },
            "total_from": {
                "A": 8
            }
        }

        tm = TransitionMatrix.from_dict(data)

        assert tm.smoothing == 0.1
        assert tm.get_count("A", "B") == 5
        assert tm.get_count("A", "C") == 3
        assert tm.total_from["A"] == 8

    def test_round_trip(self):
        """Test that to_dict -> from_dict preserves data."""
        tm1 = TransitionMatrix(smoothing=0.05)
        tm1.increment("A", "B", 10)
        tm1.increment("A", "C", 5)

        data = tm1.to_dict()
        tm2 = TransitionMatrix.from_dict(data)

        assert tm1.smoothing == tm2.smoothing
        assert tm1.get_count("A", "B") == tm2.get_count("A", "B")
        assert tm1.get_probability("A", "B") == tm2.get_probability("A", "B")

    def test_save_and_load(self):
        """Test saving to and loading from file."""
        tm1 = TransitionMatrix(smoothing=0.01)
        tm1.increment("A", "B", 10)
        tm1.increment("A", "C", 5)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            tm1.save(temp_path)
            tm2 = TransitionMatrix.load(temp_path)

            assert tm1.smoothing == tm2.smoothing
            assert tm1.get_count("A", "B") == tm2.get_count("A", "B")
            assert tm1.get_probability("A", "B") == tm2.get_probability("A", "B")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "matrix.json"
            tm.save(str(path))

            assert path.exists()

            # Verify can load
            tm2 = TransitionMatrix.load(str(path))
            assert tm2.get_count("A", "B") == 10


class TestDataFrame:
    """Test DataFrame conversion."""

    def test_to_dataframe(self):
        """Test converting to pandas DataFrame."""
        pytest.importorskip("pandas")  # Skip if pandas not installed

        tm = TransitionMatrix()
        tm.increment("A", "B", 10)
        tm.increment("A", "C", 5)
        tm.increment("B", "C", 3)

        df = tm.to_dataframe()

        assert len(df) == 3
        assert list(df.columns) == ['from_state', 'to_state', 'count', 'probability']

        # Check first row (should be sorted by from_state, to_state)
        row_ab = df[(df['from_state'] == 'A') & (df['to_state'] == 'B')].iloc[0]
        assert row_ab['count'] == 10
        assert abs(row_ab['probability'] - 10/15) < 1e-10

    def test_to_dataframe_without_pandas(self, monkeypatch):
        """Test that to_dataframe raises ImportError without pandas."""
        # Temporarily make pandas unavailable
        import src.markov.transition_matrix as tm_module
        monkeypatch.setattr(tm_module, 'PANDAS_AVAILABLE', False)

        tm = TransitionMatrix()
        tm.increment("A", "B", 10)

        with pytest.raises(ImportError, match="pandas is required"):
            tm.to_dataframe()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_matrix(self):
        """Test operations on empty matrix."""
        tm = TransitionMatrix()

        assert tm.num_states == 0
        assert tm.num_transitions == 0
        assert tm.get_count("A", "B") == 0
        assert tm.get_probability("A", "B") == 0.0
        assert tm.get_row("A") == {}
        assert tm.get_top_k("A", 5) == []

    def test_increment_zero_count(self):
        """Test that incrementing by zero has no effect."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 0)

        assert tm.get_count("A", "B") == 0
        assert tm.num_transitions == 0

    def test_increment_negative_raises(self):
        """Test that negative count raises ValueError."""
        tm = TransitionMatrix()

        with pytest.raises(ValueError, match="non-negative"):
            tm.increment("A", "B", -5)

    def test_repr(self):
        """Test string representation."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 10)
        tm.increment("B", "C", 5)

        repr_str = repr(tm)

        assert "TransitionMatrix" in repr_str
        assert "states=" in repr_str
        assert "transitions=" in repr_str
        assert "sparsity=" in repr_str

    def test_str(self):
        """Test str() representation."""
        tm = TransitionMatrix()
        tm.increment("A", "B", 10)

        str_str = str(tm)

        assert "TransitionMatrix" in str_str


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_api_endpoint_transitions(self):
        """Test realistic API endpoint transition scenario."""
        tm = TransitionMatrix(smoothing=0.001)

        # Simulate typical API usage patterns
        tm.increment("login", "profile", 80)
        tm.increment("login", "browse", 20)
        tm.increment("profile", "browse", 50)
        tm.increment("profile", "orders", 30)
        tm.increment("browse", "product", 60)
        tm.increment("browse", "search", 40)
        tm.increment("product", "cart", 70)
        tm.increment("product", "browse", 30)

        # Verify realistic probabilities
        assert abs(tm.get_probability("login", "profile") - 0.8) < 0.01
        assert abs(tm.get_probability("login", "browse") - 0.2) < 0.01

        # Check top transitions
        top_from_product = tm.get_top_k("product", k=2)
        assert top_from_product[0][0] == "cart"  # Most likely next step

        # Verify matrix properties
        # States: login, profile, browse, orders, product, search, cart = 7
        assert tm.num_states == 7
        assert tm.num_transitions == 8

    def test_large_matrix_performance(self):
        """Test performance with larger matrix (basic smoke test)."""
        tm = TransitionMatrix()

        # Create a larger matrix
        num_states = 100
        for i in range(num_states):
            for j in range(min(5, num_states)):  # Each state has up to 5 transitions
                tm.increment(f"state_{i}", f"state_{(i+j+1) % num_states}", 1)

        assert tm.num_states == num_states
        assert tm.num_transitions <= num_states * 5

        # Test that operations still work
        prob = tm.get_probability("state_0", "state_1")
        assert 0.0 <= prob <= 1.0

        top = tm.get_top_k("state_0", k=3)
        assert len(top) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

