"""Model training for overtake prediction.

Author: João Pedro Cunha
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
):
    """Train XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration

    Returns:
        Trained model
    """
    logger.info("Training XGBoost classifier...")

    # Calculate scale_pos_weight for imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=config.model.xgb_n_estimators,
        max_depth=config.model.xgb_max_depth,
        learning_rate=config.model.xgb_learning_rate,
        subsample=config.model.xgb_subsample,
        scale_pos_weight=scale_pos_weight if config.model.use_class_weights else 1.0,
        random_state=config.model.random_seed,
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    logger.info("XGBoost training complete")

    return model


def train_all_models(
    train_df: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> Dict[str, any]:
    """Train all models.

    Args:
        train_df: Training DataFrame
        config: Configuration

    Returns:
        Dictionary of trained models
    """
    logger.info("Training all models...")

    # Prepare data
    X_train, y_train = prepare_xy(train_df)

    # Get class weights
    class_weights = get_class_weights(y_train) if config.model.use_class_weights else None

    models = {}

    # Train baseline
    if config.model.train_baseline:
        models["logistic"] = train_baseline(
            X_train, y_train, class_weights, config.model.random_seed
        )

    # Train XGBoost
    if config.model.train_boosting:
        models["xgboost"] = train_xgboost(X_train, y_train, config)

    logger.info(f"Trained {len(models)} models: {list(models.keys())}")

    # Store feature names for later use
    for model_name in models:
        models[model_name].feature_names_ = list(X_train.columns)

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

    parser = argparse.ArgumentParser(description="Train overtake prediction models")
    parser.add_argument(
        "--quick", action="store_true", help="Use quick mode (fewer races)"
    )

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
    models = train_all_models(train_df, cfg)

    # Save models
    save_models(models, cfg.data.models_dir)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models trained: {list(models.keys())}")
    print(f"Models saved to: {cfg.data.models_dir}")
    print("=" * 60)
