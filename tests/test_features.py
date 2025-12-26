"""Tests for feature engineering.

Author: João Pedro Cunha
"""

import pandas as pd
import pytest

from f1overtake.features import engineer_features, select_features


def create_mock_opportunities():
    """Create synthetic overtake opportunities for testing."""
    return pd.DataFrame(
        {
            "RaceName": ["TestRace"] * 4,
            "LapNumber": [10, 10, 20, 20],
            "Driver": ["A", "B", "A", "B"],
            "DriverAhead": ["C", "A", "C", "A"],
            "Position": [2, 3, 2, 3],
            "PositionAhead": [1, 2, 1, 2],
            "Gap": [1.0, 0.5, 2.0, 1.5],
            "Overtake": [False, True, False, False],
            "LapTimeSeconds": [90.0, 90.5, 89.5, 90.2],
            "Compound": ["SOFT", "MEDIUM", "SOFT", "HARD"],
            "TyreLife": [5, 10, 15, 8],
        }
    )


def create_mock_lap_data():
    """Create synthetic full lap data."""
    return pd.DataFrame(
        {
            "RaceName": ["TestRace"] * 12,
            "LapNumber": [8, 8, 8, 9, 9, 9, 10, 10, 10, 20, 20, 20],
            "Driver": ["A", "B", "C"] * 4,
            "Position": [2, 3, 1] * 4,
            "LapTimeSeconds": [90.0, 90.5, 89.5] * 4,
            "Compound": ["SOFT"] * 12,
            "TyreLife": [1, 2, 3] * 4,
        }
    )


class TestFeatureEngineering:
    """Tests for feature engineering."""

    def test_engineer_features_basic(self):
        """Test basic feature engineering."""
        opportunities = create_mock_opportunities()
        lap_data = create_mock_lap_data()

        featured = engineer_features(opportunities, lap_data)

        assert len(featured) == len(opportunities)
        assert "Gap" in featured.columns
        assert "TyreLife" in featured.columns

    def test_gap_features(self):
        """Test gap feature engineering."""
        opportunities = create_mock_opportunities()
        lap_data = create_mock_lap_data()

        featured = engineer_features(opportunities, lap_data)

        # Should have gap bin
        assert "GapBin" in featured.columns

    def test_tire_features(self):
        """Test tire feature engineering."""
        opportunities = create_mock_opportunities()
        lap_data = create_mock_lap_data()

        featured = engineer_features(opportunities, lap_data)

        # Should have tire-related features
        assert "TireAgeDiff" in featured.columns or "AheadTyreLife" in featured.columns

    def test_select_features(self):
        """Test feature selection."""
        opportunities = create_mock_opportunities()
        lap_data = create_mock_lap_data()

        featured = engineer_features(opportunities, lap_data)
        selected = select_features(featured)

        # Should have target column
        assert "Overtake" in selected.columns

        # Should have metadata
        assert "Driver" in selected.columns
        assert "RaceName" in selected.columns
