"""Model evaluation metrics and utilities.

Author: João Pedro Cunha
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from f1overtake.split import prepare_xy

logger = logging.getLogger(__name__)


def evaluate_model(model, test_df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate model on test set.

    Args:
        model: Trained model
        test_df: Test DataFrame
        threshold: Decision threshold for classification

    Returns:
        Dictionary of evaluation metrics
    """
    # Prepare data
    X_test, y_test = prepare_xy(test_df)

    # Align features if needed
    if hasattr(model, "feature_names_"):
        X_test = X_test[model.feature_names_]

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "brier_score": brier_score_loss(y_test, y_pred_proba),
    }

    # PR AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    metrics["pr_auc"] = auc(recall, precision)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()

    return metrics


def evaluate_all_models(
    models: Dict[str, any], test_df: pd.DataFrame, threshold: float = 0.5
) -> pd.DataFrame:
    """Evaluate all models and return comparison DataFrame.

    Args:
        models: Dictionary of models
        test_df: Test DataFrame
        threshold: Decision threshold

    Returns:
        DataFrame with evaluation metrics for each model
    """
    logger.info("Evaluating all models...")

    results = []

    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}")
        metrics = evaluate_model(model, test_df, threshold)
        metrics["model"] = model_name
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("model")

    logger.info(f"Evaluation complete for {len(models)} models")

    return results_df


def get_calibration_data(model, test_df: pd.DataFrame, n_bins: int = 10) -> Dict:
    """Get calibration curve data.

    Args:
        model: Trained model
        test_df: Test DataFrame
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration data
    """
    X_test, y_test = prepare_xy(test_df)

    if hasattr(model, "feature_names_"):
        X_test = X_test[model.feature_names_]

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Bin predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Calculate actual frequencies per bin
    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_sums[i] = y_test[mask].sum()
            bin_counts[i] = mask.sum()

    bin_means = np.where(bin_counts > 0, bin_sums / bin_counts, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    return {
        "bin_centers": bin_centers,
        "bin_means": bin_means,
        "bin_counts": bin_counts,
        "predicted_probs": y_pred_proba,
        "actual_labels": y_test,
    }


def get_feature_importance(model, top_n: int = 10) -> pd.DataFrame:
    """Get feature importance from model.

    Args:
        model: Trained model
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importances
    """
    if hasattr(model, "feature_importances_"):
        # Tree-based model (XGBoost)
        importances = model.feature_importances_
        feature_names = model.feature_names_ if hasattr(model, "feature_names_") else None
    elif hasattr(model, "coef_"):
        # Linear model (Logistic Regression)
        importances = np.abs(model.coef_[0])
        feature_names = model.feature_names_ if hasattr(model, "feature_names_") else None
    else:
        # Try to get from calibrated model
        if hasattr(model, "base_estimator"):
            return get_feature_importance(model.base_estimator, top_n)
        logger.warning("Model does not support feature importance")
        return pd.DataFrame()

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    return importance_df.head(top_n)
