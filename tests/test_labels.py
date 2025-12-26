"""Tests for overtake labeling logic.

Author: João Pedro Cunha
"""

import pandas as pd
import pytest

from f1overtake.config import LabelConfig
from f1overtake.labels import create_overtake_labels


def create_mock_lap_data():
    """Create synthetic lap data for testing."""
    return pd.DataFrame(
        {
            "RaceName": ["TestRace"] * 10,
            "LapNumber": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "Driver": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "Position": [1, 2, 1, 2, 2, 1, 2, 1, 2, 1],  # B overtakes A on lap 3
            "LapTimeSeconds": [90.0] * 10,
            "Compound": ["SOFT"] * 10,
            "TyreLife": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "IsPitLap": [False] * 10,
        }
    )


class TestOvertakeLabeling:
    """Tests for overtake labeling."""

    def test_create_labels_basic(self):
        """Test basic label creation."""
        lap_data = create_mock_lap_data()
        config = LabelConfig(
            lookahead_laps=1, max_gap=10.0, exclude_pit_laps=False, exclude_outliers=False
        )

        labels = create_overtake_labels(lap_data, config)

        assert len(labels) > 0
        assert "Overtake" in labels.columns
        assert "Driver" in labels.columns

    def test_overtake_detection(self):
        """Test that overtakes are correctly detected."""
        lap_data = create_mock_lap_data()
        config = LabelConfig(
            lookahead_laps=1, max_gap=10.0, exclude_pit_laps=False, exclude_outliers=False
        )

        labels = create_overtake_labels(lap_data, config)

        # Driver B was behind A on lap 2 (position 2 vs 1), ahead on lap 3 (position 1 vs 2)
        # So driver B should have an overtake label on lap 2
        driver_b_labels = labels[labels["Driver"] == "B"]
        overtakes = driver_b_labels[driver_b_labels["Overtake"] == True]

        # Should have at least one overtake
        assert len(overtakes) > 0

    def test_pit_lap_exclusion(self):
        """Test that pit laps are excluded."""
        lap_data = create_mock_lap_data()

        # Mark lap 3 as pit lap
        lap_data.loc[lap_data["LapNumber"] == 3, "IsPitLap"] = True

        config = LabelConfig(
            lookahead_laps=1, max_gap=10.0, exclude_pit_laps=True, exclude_outliers=False
        )

        labels = create_overtake_labels(lap_data, config)

        # Lap 2 opportunities should be excluded (because lap 3 has pit)
        lap_2_labels = labels[labels["LapNumber"] == 2]

        # Should have fewer opportunities
        assert len(lap_2_labels) == 0

    def test_no_data_returns_empty(self):
        """Test that empty input returns empty DataFrame."""
        lap_data = pd.DataFrame()
        config = LabelConfig()

        labels = create_overtake_labels(lap_data, config)

        assert len(labels) == 0
