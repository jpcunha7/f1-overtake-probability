"""Tests for temporal features.

Author: João Pedro Cunha
"""

import numpy as np
import pandas as pd
import pytest

from f1overtake.config import FeatureConfig
from f1overtake.features import _add_temporal_features


@pytest.fixture
def sample_opportunities():
    """Create sample overtake opportunities with sequential laps."""
    data = []
    for lap in range(1, 11):
        data.append({
            "RaceName": "Bahrain",
            "LapNumber": lap,
            "Driver": "VER",
            "DriverAhead": "PER",
            "Gap": 2.0 - (lap * 0.1),  # Gap closing over time
            "RelativePace": 0.5,
            "Position": 2,
        })

    return pd.DataFrame(data)


@pytest.fixture
def sample_lap_data():
    """Create sample lap data."""
    data = []
    for lap in range(1, 11):
        data.append({
            "RaceName": "Bahrain",
            "LapNumber": lap,
            "Driver": "VER",
            "LapTimeSeconds": 90.0 + np.random.uniform(-1, 1),
        })
        data.append({
            "RaceName": "Bahrain",
            "LapNumber": lap,
            "Driver": "PER",
            "LapTimeSeconds": 90.5 + np.random.uniform(-1, 1),
        })

    return pd.DataFrame(data)


def test_add_temporal_features_basic(sample_opportunities, sample_lap_data):
    """Test basic temporal feature addition."""
    config = FeatureConfig(
        enable_temporal_features=True,
        lagged_gap_laps=[1, 2],
        closing_rate_window=3,
    )

    result = _add_temporal_features(
        sample_opportunities.copy(), sample_lap_data, config
    )

    # Should have lagged gap columns
    assert "Gap_L1" in result.columns
    assert "Gap_L2" in result.columns

    # Should have closing rate
    assert "ClosingRate" in result.columns

    # Should have rolling pace delta
    assert "RollingPaceDelta" in result.columns


def test_lagged_gap_values(sample_opportunities, sample_lap_data):
    """Test that lagged gap values are correct."""
    config = FeatureConfig(
        enable_temporal_features=True,
        lagged_gap_laps=[1],
        closing_rate_window=3,
    )

    result = _add_temporal_features(
        sample_opportunities.copy(), sample_lap_data, config
    )

    # For lap 3, Gap_L1 should be the gap from lap 2
    lap_3 = result[result["LapNumber"] == 3].iloc[0]
    lap_2_gap = sample_opportunities[sample_opportunities["LapNumber"] == 2].iloc[0]["Gap"]

    if pd.notna(lap_3["Gap_L1"]):
        assert np.isclose(lap_3["Gap_L1"], lap_2_gap, rtol=0.01)


def test_closing_rate_calculation(sample_opportunities, sample_lap_data):
    """Test closing rate calculation."""
    config = FeatureConfig(
        enable_temporal_features=True,
        lagged_gap_laps=[2],
        closing_rate_window=3,
    )

    result = _add_temporal_features(
        sample_opportunities.copy(), sample_lap_data, config
    )

    # Closing rate should be (old_gap - current_gap) / lag
    # Since gap is decreasing, closing rate should be positive
    lap_5 = result[result["LapNumber"] == 5].iloc[0]

    if pd.notna(lap_5["ClosingRate"]):
        # Gap is decreasing (2.0, 1.9, 1.8, ...) so closing rate should be positive
        assert lap_5["ClosingRate"] > 0


def test_missing_lags_at_start(sample_opportunities, sample_lap_data):
    """Test that missing lags at stint start are handled properly."""
    config = FeatureConfig(
        enable_temporal_features=True,
        lagged_gap_laps=[1, 2],
        closing_rate_window=3,
    )

    result = _add_temporal_features(
        sample_opportunities.copy(), sample_lap_data, config
    )

    # Lap 1 should have NaN for Gap_L1 (no previous lap)
    lap_1 = result[result["LapNumber"] == 1].iloc[0]
    assert pd.isna(lap_1["Gap_L1"])

    # Lap 2 should have value for Gap_L1 but NaN for Gap_L2
    lap_2 = result[result["LapNumber"] == 2].iloc[0]
    assert pd.notna(lap_2["Gap_L1"])
    assert pd.isna(lap_2["Gap_L2"])


def test_temporal_features_with_no_lags():
    """Test temporal features with empty lagged_gap_laps."""
    config = FeatureConfig(
        enable_temporal_features=True,
        lagged_gap_laps=[],
        closing_rate_window=3,
    )

    df = pd.DataFrame({
        "RaceName": ["Bahrain"],
        "LapNumber": [5],
        "Driver": ["VER"],
        "DriverAhead": ["PER"],
        "Gap": [1.5],
    })

    lap_data = pd.DataFrame()

    result = _add_temporal_features(df, lap_data, config)

    # Should not have ClosingRate if no lagged gaps
    # But should still have RollingPaceDelta
    assert "RollingPaceDelta" in result.columns
