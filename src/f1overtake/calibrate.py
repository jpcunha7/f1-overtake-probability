"""Probability calibration for overtake prediction models.

Author: João Pedro Cunha

Calibration ensures that predicted probabilities align with actual frequencies.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from f1overtake.config import Config, DEFAULT_CONFIG
from f1overtake.split import prepare_xy

logger = logging.getLogger(__name__)


def calibrate_model(model, X_cal: pd.DataFrame, y_cal: pd.Series, method: str = "isotonic"):
    """Calibrate a trained model.

    Args:
        model: Trained classifier
        X_cal: Calibration features
        y_cal: Calibration labels
        method: Calibration method ('sigmoid' or 'isotonic')

    Returns:
        Calibrated model
    """
    logger.info(f"Calibrating model using {method} method...")

    # Use cv=2 for minimal cross-validation during calibration
    # This retrains the model on subsets but provides proper calibration
    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=2,  # Use 2-fold CV for calibration
        ensemble=False,  # Don't create ensemble, just calibrate
    )

    calibrated.fit(X_cal, y_cal)

    logger.info("Calibration complete")

    return calibrated


def calibrate_all_models(
    models: Dict[str, any],
    cal_df: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> Dict[str, any]:
    """Calibrate all models.

    Args:
        models: Dictionary of trained models
        cal_df: Calibration DataFrame (typically validation set)
        config: Configuration

    Returns:
        Dictionary of calibrated models
    """
    logger.info("Calibrating all models...")

    # Prepare calibration data
    X_cal, y_cal = prepare_xy(cal_df)

    calibrated_models = {}

    for model_name, model in models.items():
        # Ensure feature alignment
        if hasattr(model, "feature_names_"):
            X_cal_aligned = X_cal[model.feature_names_]
        else:
            X_cal_aligned = X_cal

        calibrated = calibrate_model(
            model, X_cal_aligned, y_cal, method=config.model.calibration_method
        )

        # Preserve feature names
        calibrated.feature_names_ = list(X_cal_aligned.columns)

        calibrated_models[f"{model_name}_calibrated"] = calibrated

        logger.info(f"Calibrated {model_name}")

    return calibrated_models
