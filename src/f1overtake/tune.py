"""Hyperparameter tuning using Optuna with GroupKFold cross-validation.

Author: João Pedro Cunha

This module implements Bayesian optimization for XGBoost hyperparameters using Optuna.
Key features:
- GroupKFold cross-validation by race to prevent data leakage
- Optimizes for PR AUC (better for imbalanced classification)
- Early stopping to prevent overfitting
- Saves best parameters to JSON for reproducibility
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from f1overtake.config import Config, DEFAULT_CONFIG
from f1overtake.split import prepare_xy

logger = logging.getLogger(__name__)


def pr_auc_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate Precision-Recall AUC score.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        PR AUC score
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


def objective(
    trial,
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    config: Config,
) -> float:
    """Optuna objective function for hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X: Feature matrix
        y: Target vector
        groups: Group labels for GroupKFold (race names)
        config: Configuration object

    Returns:
        Mean PR AUC across folds
    """
    # Define hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "random_state": config.model.random_seed,
        "eval_metric": "logloss",
    }

    # GroupKFold cross-validation
    gkf = GroupKFold(n_splits=config.model.optuna_cv_folds)
    pr_auc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Calculate scale_pos_weight for imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        # Train model
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight if config.model.use_class_weights else 1.0,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        pr_auc = pr_auc_score(y_val, y_pred_proba)
        pr_auc_scores.append(pr_auc)

        logger.debug(f"Fold {fold_idx + 1}: PR AUC = {pr_auc:.4f}")

    mean_pr_auc = np.mean(pr_auc_scores)
    logger.debug(f"Trial {trial.number}: Mean PR AUC = {mean_pr_auc:.4f}")

    return mean_pr_auc


def tune_xgboost(
    train_df: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    save_path: Optional[str] = None,
) -> Dict:
    """Tune XGBoost hyperparameters using Optuna.

    Args:
        train_df: Training DataFrame with features and target
        config: Configuration object
        save_path: Optional path to save best parameters (JSON)

    Returns:
        Dictionary with best hyperparameters
    """
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Install with: pip install optuna")
        raise

    logger.info("Starting Optuna hyperparameter tuning...")
    logger.info(f"Number of trials: {config.model.optuna_n_trials}")
    logger.info(f"CV folds: {config.model.optuna_cv_folds}")
    logger.info(f"Timeout: {config.model.optuna_timeout}s")

    # Prepare data
    X, y = prepare_xy(train_df)

    # Create groups for GroupKFold (use race names)
    groups = train_df["RaceName"].values

    # Handle categorical features
    if X.select_dtypes(include=["object", "category"]).shape[1] > 0:
        logger.info("One-hot encoding categorical features...")
        X = pd.get_dummies(X, drop_first=True)

    logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Positive class rate: {y.mean()*100:.2f}%")
    logger.info(f"Number of races: {len(np.unique(groups))}")

    # Create study
    study = optuna.create_study(
        direction="maximize",  # Maximize PR AUC
        study_name="f1_overtake_xgboost_tuning",
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, groups, config),
        n_trials=config.model.optuna_n_trials,
        timeout=config.model.optuna_timeout,
        show_progress_bar=True,
    )

    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value

    logger.info("=" * 60)
    logger.info("OPTUNA TUNING RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best PR AUC: {best_score:.4f}")
    logger.info("Best parameters:")
    for param, value in best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info("=" * 60)

    # Calculate baseline PR AUC (random predictor)
    baseline_pr_auc = y.mean()
    improvement = best_score / baseline_pr_auc

    logger.info(f"Baseline PR AUC (random): {baseline_pr_auc:.4f}")
    logger.info(f"Improvement over baseline: {improvement:.2f}x")

    # Check if we meet target
    if best_score >= config.model.target_pr_auc_multiplier * baseline_pr_auc:
        logger.info(f"Target achieved: PR AUC >= {config.model.target_pr_auc_multiplier}x baseline")
    else:
        logger.warning(
            f"Target not achieved: PR AUC < {config.model.target_pr_auc_multiplier}x baseline"
        )

    # Save parameters if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        params_to_save = {
            "best_params": best_params,
            "best_pr_auc": best_score,
            "baseline_pr_auc": baseline_pr_auc,
            "improvement_factor": improvement,
            "n_trials": len(study.trials),
            "cv_folds": config.model.optuna_cv_folds,
        }

        with open(save_path, "w") as f:
            json.dump(params_to_save, f, indent=2)

        logger.info(f"Saved best parameters to {save_path}")

    return best_params


def load_best_params(params_path: str) -> Dict:
    """Load best parameters from JSON file.

    Args:
        params_path: Path to JSON file with parameters

    Returns:
        Dictionary with best hyperparameters
    """
    params_path = Path(params_path)

    if not params_path.exists():
        logger.warning(f"Parameters file not found: {params_path}")
        return {}

    with open(params_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded best parameters from {params_path}")
    logger.info(f"Best PR AUC: {data.get('best_pr_auc', 'N/A')}")

    return data.get("best_params", {})


if __name__ == "__main__":
    """Command-line interface for hyperparameter tuning."""
    import argparse

    from f1overtake.build_dataset import build_dataset
    from f1overtake.split import split_by_race

    parser = argparse.ArgumentParser(description="Tune XGBoost hyperparameters with Optuna")
    parser.add_argument(
        "--quick", action="store_true", help="Use quick mode (fewer races)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5, help="Number of CV folds"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Timeout in seconds"
    )
    parser.add_argument(
        "--output", type=str, default="models/best_params.json", help="Output path for best parameters"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Select config
    from f1overtake.config import QUICK_CONFIG

    cfg = QUICK_CONFIG if args.quick else DEFAULT_CONFIG

    # Override Optuna settings from command line
    cfg.model.optuna_n_trials = args.n_trials
    cfg.model.optuna_cv_folds = args.cv_folds
    cfg.model.optuna_timeout = args.timeout

    # Build dataset
    logger.info("Building dataset...")
    dataset = build_dataset(cfg)

    # Split (only use training data for tuning)
    logger.info("Splitting dataset...")
    train_df, test_df = split_by_race(
        dataset, test_size=cfg.model.test_size, random_seed=cfg.model.random_seed
    )

    # Tune
    best_params = tune_xgboost(train_df, cfg, save_path=args.output)

    print("\n" + "=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"Best parameters saved to: {args.output}")
    print("=" * 60)
