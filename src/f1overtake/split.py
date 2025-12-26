"""Train/test splitting with leak-age prevention.

Author: João Pedro Cunha

SPLITTING STRATEGY:
To avoid data leakage, we split by race weekend (not randomly).
This ensures the model is tested on completely unseen races.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from f1overtake.config import ModelConfig

logger = logging.getLogger(__name__)


def split_by_race(
    dataset: pd.DataFrame, test_size: float = 0.3, random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset by race to avoid leakage.

    Args:
        dataset: Full dataset with RaceName column
        test_size: Fraction of races to use for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting dataset by race (test_size={test_size})")

    # Get unique races
    unique_races = dataset["RaceName"].unique()
    logger.info(f"Total races: {len(unique_races)}")

    # Split races
    train_races, test_races = train_test_split(
        unique_races, test_size=test_size, random_state=random_seed
    )

    logger.info(f"Train races ({len(train_races)}): {list(train_races)}")
    logger.info(f"Test races ({len(test_races)}): {list(test_races)}")

    # Split dataset
    train_df = dataset[dataset["RaceName"].isin(train_races)].copy()
    test_df = dataset[dataset["RaceName"].isin(test_races)].copy()

    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    logger.info(
        f"Train overtake rate: {train_df['Overtake'].mean()*100:.1f}%, "
        f"Test overtake rate: {test_df['Overtake'].mean()*100:.1f}%"
    )

    return train_df, test_df


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = "Overtake",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare X and y for modeling.

    Args:
        df: Dataset DataFrame
        feature_cols: List of feature columns (if None, auto-detect)
        target_col: Target column name

    Returns:
        Tuple of (X, y)
    """
    if feature_cols is None:
        # Auto-detect: all columns except target and metadata
        metadata_cols = ["RaceName", "Driver", "DriverAhead", "LapNumber"]
        feature_cols = [
            col
            for col in df.columns
            if col not in [target_col] + metadata_cols
        ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        logger.info(f"One-hot encoding categorical columns: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    logger.info(f"Prepared X: {X.shape}, y: {y.shape}")
    logger.info(f"Features: {list(X.columns)}")

    return X, y


def get_class_weights(y: pd.Series) -> dict:
    """Calculate class weights for imbalanced data.

    Args:
        y: Target series

    Returns:
        Dictionary of class weights
    """
    pos_count = y.sum()
    neg_count = len(y) - pos_count

    # Inverse frequency weighting
    total = len(y)
    weight_pos = total / (2 * pos_count) if pos_count > 0 else 1.0
    weight_neg = total / (2 * neg_count) if neg_count > 0 else 1.0

    weights = {0: weight_neg, 1: weight_pos}

    logger.info(f"Class weights: {weights}")

    return weights
