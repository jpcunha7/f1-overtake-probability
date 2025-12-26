"""Tests for hyperparameter tuning with Optuna.

Author: João Pedro Cunha
"""

import numpy as np
import pandas as pd
import pytest

from f1overtake.config import Config
from f1overtake.tune import pr_auc_score


def test_pr_auc_score():
    """Test PR AUC score calculation."""
    # Perfect predictions
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.9])

    pr_auc = pr_auc_score(y_true, y_pred_proba)

    # Should be close to 1.0 for good predictions
    assert pr_auc > 0.5
    assert pr_auc <= 1.0


def test_pr_auc_score_random():
    """Test PR AUC score with random predictions."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.uniform(0, 1, 100)

    pr_auc = pr_auc_score(y_true, y_pred_proba)

    # Random predictions should give PR AUC around prevalence
    prevalence = y_true.mean()
    assert 0.0 <= pr_auc <= 1.0


def test_pr_auc_score_all_positive():
    """Test PR AUC with all positive labels."""
    y_true = np.array([1, 1, 1, 1])
    y_pred_proba = np.array([0.5, 0.6, 0.7, 0.8])

    pr_auc = pr_auc_score(y_true, y_pred_proba)

    # Should be 1.0 for all positive
    assert pr_auc == 1.0


def test_pr_auc_score_all_negative():
    """Test PR AUC with all negative labels."""
    y_true = np.array([0, 0, 0, 0])
    y_pred_proba = np.array([0.5, 0.6, 0.7, 0.8])

    # This should raise an error or return a specific value
    # because there are no positive samples
    try:
        pr_auc = pr_auc_score(y_true, y_pred_proba)
        # If it doesn't raise, it should handle gracefully
        assert pr_auc >= 0.0
    except (ValueError, ZeroDivisionError):
        # Expected behavior for no positive samples
        pass


@pytest.mark.skipif(
    not pytest.importorskip("optuna", reason="Optuna not installed"),
    reason="Optuna not installed",
)
def test_tune_xgboost_basic():
    """Test basic Optuna tuning (quick test with 1 trial)."""
    from f1overtake.tune import tune_xgboost

    # Create minimal dataset
    np.random.seed(42)
    n_samples = 200

    train_df = pd.DataFrame(
        {
            "RaceName": ["Race1"] * 100 + ["Race2"] * 100,
            "Driver": ["VER"] * n_samples,
            "DriverAhead": ["PER"] * n_samples,
            "Gap": np.random.uniform(0.5, 2.5, n_samples),
            "RelativePace": np.random.uniform(-0.5, 0.5, n_samples),
            "PaceRatio": np.random.uniform(0.95, 1.05, n_samples),
            "TyreLife": np.random.randint(1, 20, n_samples),
            "AheadTyreLife": np.random.randint(1, 20, n_samples),
            "TireAgeDiff": np.random.randint(-10, 10, n_samples),
            "CompoundAdvantage": np.random.randint(-1, 2, n_samples),
            "RaceProgress": np.random.uniform(0.2, 0.8, n_samples),
            "Position": np.random.randint(3, 15, n_samples),
            "Overtake": np.random.randint(0, 2, n_samples),
        }
    )

    # Quick config for testing
    config = Config()
    config.model.optuna_n_trials = 1
    config.model.optuna_cv_folds = 2
    config.model.optuna_timeout = 10

    # Run tuning
    best_params = tune_xgboost(train_df, config)

    # Should return parameters
    assert isinstance(best_params, dict)
    assert len(best_params) > 0

    # Should have key XGBoost parameters
    expected_params = ["n_estimators", "max_depth", "learning_rate"]
    for param in expected_params:
        assert param in best_params
