"""Model training for overtake prediction.

Author: João Pedro Cunha
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

from f1overtake.config import Config, DEFAULT_CONFIG
from f1overtake.split import get_class_weights, prepare_xy

logger = logging.getLogger(__name__)


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weights: dict = None,
    random_seed: int = 42,
):
    """Train logistic regression baseline.

    Args:
        X_train: Training features
        y_train: Training labels
        class_weights: Class weights for imbalance
        random_seed: Random seed

    Returns:
        Trained model
    """
    logger.info("Training Logistic Regression baseline...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=random_seed,
        class_weight=class_weights if class_weights else None,
    )

    model.fit(X_train, y_train)

    logger.info("Logistic Regression training complete")

    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Config = DEFAULT_CONFIG,
    best_params: Optional[Dict] = None,
):
    """Train XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration
        best_params: Optional best hyperparameters from tuning

    Returns:
        Trained model
    """
    logger.info("Training XGBoost classifier...")

    # Calculate scale_pos_weight for imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    logger.info(f"Class distribution: {neg_count} negative, {pos_count} positive")
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Use tuned parameters if available, otherwise use config
    if best_params:
        logger.info("Using tuned hyperparameters from Optuna")
        params = {
            "n_estimators": best_params.get("n_estimators", config.model.xgb_n_estimators),
            "max_depth": best_params.get("max_depth", config.model.xgb_max_depth),
            "learning_rate": best_params.get("learning_rate", config.model.xgb_learning_rate),
            "subsample": best_params.get("subsample", config.model.xgb_subsample),
            "colsample_bytree": best_params.get(
                "colsample_bytree", config.model.xgb_colsample_bytree
            ),
            "min_child_weight": best_params.get(
                "min_child_weight", config.model.xgb_min_child_weight
            ),
            "gamma": best_params.get("gamma", config.model.xgb_gamma),
        }
    else:
        logger.info("Using default hyperparameters from config")
        params = {
            "n_estimators": config.model.xgb_n_estimators,
            "max_depth": config.model.xgb_max_depth,
            "learning_rate": config.model.xgb_learning_rate,
            "subsample": config.model.xgb_subsample,
            "colsample_bytree": config.model.xgb_colsample_bytree,
            "min_child_weight": config.model.xgb_min_child_weight,
            "gamma": config.model.xgb_gamma,
        }

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight if config.model.use_class_weights else 1.0,
        random_state=config.model.random_seed,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train, verbose=False)

    logger.info("XGBoost training complete")

    return model


def train_all_models(
    train_df: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    use_tuning: bool = False,
) -> Dict[str, any]:
    """Train all models.

    Args:
        train_df: Training DataFrame
        config: Configuration
        use_tuning: Whether to use Optuna hyperparameter tuning

    Returns:
        Dictionary of trained models
    """
    logger.info("Training all models...")
    logger.info(f"Use Optuna tuning: {use_tuning or config.model.use_optuna}")

    # Prepare data
    X_train, y_train = prepare_xy(train_df)

    # Get class weights
    class_weights = get_class_weights(y_train) if config.model.use_class_weights else None

    # Optuna tuning if enabled
    best_params = None
    if use_tuning or config.model.use_optuna:
        logger.info("Running Optuna hyperparameter tuning...")
        try:
            from f1overtake.tune import tune_xgboost as run_tuning

            best_params = run_tuning(
                train_df,
                config,
                save_path=str(Path(config.data.models_dir) / "best_params.json"),
            )
        except ImportError:
            logger.warning("Optuna not installed. Skipping tuning.")
        except Exception as e:
            logger.error(f"Tuning failed: {e}. Using default parameters.")

    models = {}

    # Train baseline
    if config.model.train_baseline:
        models["logistic"] = train_baseline(
            X_train, y_train, class_weights, config.model.random_seed
        )

    # Train XGBoost
    if config.model.train_boosting:
        models["xgboost"] = train_xgboost(X_train, y_train, config, best_params)

    logger.info(f"Trained {len(models)} models: {list(models.keys())}")

    # Store feature names for later use
    for model_name in models:
        models[model_name].feature_names_ = list(X_train.columns)

    # Log training metrics
    logger.info("=" * 60)
    logger.info("TRAINING METRICS")
    logger.info("=" * 60)
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_train)[:, 1]
        roc_auc = roc_auc_score(y_train, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_train, y_pred_proba)
        pr_auc = auc(recall, precision)

        logger.info(f"{model_name}:")
        logger.info(f"  ROC AUC (train): {roc_auc:.4f}")
        logger.info(f"  PR AUC (train): {pr_auc:.4f}")

        # Check if we meet targets
        baseline_pr_auc = y_train.mean()
        if roc_auc >= config.model.target_roc_auc:
            logger.info(f"  ROC AUC target achieved (>= {config.model.target_roc_auc})")
        else:
            logger.warning(f"  ROC AUC target not met (< {config.model.target_roc_auc})")

        if pr_auc >= config.model.target_pr_auc_multiplier * baseline_pr_auc:
            logger.info(
                f"  PR AUC target achieved (>= {config.model.target_pr_auc_multiplier}x baseline)"
            )
        else:
            logger.warning(
                f"  PR AUC target not met (< {config.model.target_pr_auc_multiplier}x baseline)"
            )

    logger.info("=" * 60)

    return models


def save_models(models: Dict[str, any], models_dir: str = "models") -> None:
    """Save trained models to disk.

    Args:
        models: Dictionary of trained models
        models_dir: Directory to save models
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        model_file = models_path / f"{model_name}.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved {model_name} to {model_file}")


def load_models(models_dir: str = "models") -> Dict[str, any]:
    """Load trained models from disk.

    Args:
        models_dir: Directory containing saved models

    Returns:
        Dictionary of loaded models
    """
    models_path = Path(models_dir)
    models = {}

    for model_file in models_path.glob("*.pkl"):
        model_name = model_file.stem
        with open(model_file, "rb") as f:
            models[model_name] = pickle.load(f)
        logger.info(f"Loaded {model_name} from {model_file}")

    return models


if __name__ == "__main__":
    """Command-line interface for training models."""
    import argparse

    from f1overtake.build_dataset import build_dataset
    from f1overtake.split import split_by_race

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train overtake prediction models")
    parser.add_argument("--quick", action="store_true", help="Use quick mode (fewer races)")
    parser.add_argument("--tune", action="store_true", help="Use Optuna hyperparameter tuning")

    args = parser.parse_args()

    # Select config
    from f1overtake.config import QUICK_CONFIG

    cfg = QUICK_CONFIG if args.quick else DEFAULT_CONFIG

    # Build dataset
    dataset = build_dataset(cfg)

    # Split
    train_df, test_df = split_by_race(
        dataset, test_size=cfg.model.test_size, random_seed=cfg.model.random_seed
    )

    # Train models
    models = train_all_models(train_df, cfg, use_tuning=args.tune)

    # Save models
    save_models(models, cfg.data.models_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models trained: {list(models.keys())}")
    print(f"Models saved to: {cfg.data.models_dir}")
    if args.tune:
        print(f"Best parameters saved to: {cfg.data.models_dir}/best_params.json")
    print("=" * 60)
