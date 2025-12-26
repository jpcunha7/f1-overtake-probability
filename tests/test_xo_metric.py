"""Tests for xO metric calculation.

Author: João Pedro Cunha
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from f1overtake.config import XOConfig
from f1overtake.xo_metric import calculate_xo, create_xo_leaderboard


@pytest.fixture
def sample_df():
    """Create sample overtake opportunities DataFrame."""
    return pd.DataFrame(
        {
            "RaceName": ["Bahrain"] * 20,
            "Driver": ["VER"] * 10 + ["HAM"] * 10,
            "DriverAhead": ["PER"] * 10 + ["RUS"] * 10,
            "Gap": np.random.uniform(0.5, 2.5, 20),
            "RelativePace": np.random.uniform(-0.5, 0.5, 20),
            "PaceRatio": np.random.uniform(0.95, 1.05, 20),
            "TyreLife": np.random.randint(1, 20, 20),
            "AheadTyreLife": np.random.randint(1, 20, 20),
            "TireAgeDiff": np.random.randint(-10, 10, 20),
            "CompoundAdvantage": np.random.randint(-1, 2, 20),
            "RaceProgress": np.random.uniform(0.2, 0.8, 20),
            "Position": np.random.randint(3, 15, 20),
            "Overtake": [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        }
    )


@pytest.fixture
def simple_model(sample_df):
    """Create a simple trained model."""
    from f1overtake.split import prepare_xy

    X, y = prepare_xy(sample_df)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    model.feature_names_ = list(X.columns)
    return model


def test_calculate_xo(sample_df, simple_model):
    """Test xO calculation."""
    config = XOConfig(min_opportunities=3)
    xo_df = calculate_xo(sample_df, simple_model, config)

    # Should have results for both drivers
    assert len(xo_df) >= 1
    assert "xO" in xo_df.columns
    assert "ActualOvertakes" in xo_df.columns
    assert "Delta" in xo_df.columns

    # xO should be positive
    assert (xo_df["xO"] >= 0).all()

    # Delta should be actual - xO
    assert np.allclose(xo_df["Delta"], xo_df["ActualOvertakes"] - xo_df["xO"])


def test_calculate_xo_empty_df():
    """Test xO calculation with empty DataFrame."""
    from sklearn.linear_model import LogisticRegression

    empty_df = pd.DataFrame()
    model = LogisticRegression()

    config = XOConfig()
    xo_df = calculate_xo(empty_df, model, config)

    assert len(xo_df) == 0


def test_create_xo_leaderboard(sample_df, simple_model):
    """Test xO leaderboard creation."""
    config = XOConfig(min_opportunities=3)
    xo_df = calculate_xo(sample_df, simple_model, config)

    # Test race-specific leaderboard
    leaderboard = create_xo_leaderboard(xo_df, race_name="Bahrain")
    assert len(leaderboard) >= 1
    assert "Driver" in leaderboard.columns

    # Test aggregate leaderboard
    agg_leaderboard = create_xo_leaderboard(xo_df, race_name=None)
    assert len(agg_leaderboard) >= 1


def test_xo_min_opportunities(sample_df, simple_model):
    """Test that min_opportunities filter works."""
    # Set high threshold
    config = XOConfig(min_opportunities=50)
    xo_df = calculate_xo(sample_df, simple_model, config)

    # Should have no results
    assert len(xo_df) == 0


def test_xo_per_opportunity(sample_df, simple_model):
    """Test xO per opportunity calculation."""
    config = XOConfig(min_opportunities=3)
    xo_df = calculate_xo(sample_df, simple_model, config)

    if len(xo_df) > 0:
        # xO per opportunity should be between 0 and 1
        assert (xo_df["xO_per_Opportunity"] >= 0).all()
        assert (xo_df["xO_per_Opportunity"] <= 1).all()

        # Should equal xO / Opportunities
        assert np.allclose(xo_df["xO_per_Opportunity"], xo_df["xO"] / xo_df["Opportunities"])
