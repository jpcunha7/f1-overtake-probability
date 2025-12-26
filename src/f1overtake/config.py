"""Configuration management for F1 Overtake Probability Model.

Author: João Pedro Cunha
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data loading and caching configuration."""

    cache_dir: str = "cache"
    dataset_dir: str = "datasets"
    models_dir: str = "models"
    outputs_dir: str = "outputs"

    # Races to include in dataset
    year: int = 2024
    race_names: List[str] = field(
        default_factory=lambda: [
            "Bahrain",
            "Saudi Arabia",
            "Australia",
            "Japan",
            "China",
            "Miami",
            "Emilia Romagna",
            "Monaco",
            "Canada",
            "Spain",
            "Austria",
            "Great Britain",
            "Hungary",
            "Belgium",
            "Netherlands",
            "Italy",
            "Azerbaijan",
            "Singapore",
            "United States",
            "Mexico",
            "Brazil",
            "Las Vegas",
            "Qatar",
            "Abu Dhabi",
        ]
    )

    # Quick mode for testing (fewer races)
    quick_mode: bool = False

    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.cache_dir, self.dataset_dir, self.models_dir, self.outputs_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class LabelConfig:
    """Overtake labeling configuration."""

    # How many laps ahead to check for position change
    lookahead_laps: int = 1

    # Minimum gap (seconds) to consider as an overtake opportunity
    # (ignore cars very far apart)
    max_gap: float = 3.0

    # Exclude laps where either driver pits
    exclude_pit_laps: bool = True

    # Exclude outlier lap times (formation laps, SC, etc.)
    exclude_outliers: bool = True
    outlier_threshold: float = 3.0  # std devs from median

    # Minimum stint length to consider
    min_stint_length: int = 3


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    # Rolling window sizes for pace calculations
    pace_window: int = 3

    # Normalize features
    normalize: bool = True


@dataclass
class ModelConfig:
    """Model training configuration."""

    # Random seed for reproducibility
    random_seed: int = 42

    # Model types to train
    train_baseline: bool = True  # Logistic Regression
    train_boosting: bool = True  # XGBoost

    # XGBoost hyperparameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8

    # Class imbalance handling
    use_class_weights: bool = True

    # Calibration method
    calibration_method: str = "isotonic"  # or 'sigmoid'

    # Test split
    test_size: float = 0.3  # 30% of races for testing


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Threshold for binary classification
    decision_threshold: float = 0.5

    # Generate calibration curves
    n_bins: int = 10


@dataclass
class Config:
    """Master configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        """Validate configuration."""
        if self.labels.lookahead_laps < 1:
            raise ValueError("lookahead_laps must be >= 1")
        if self.model.test_size <= 0 or self.model.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")

        logger.info("Configuration initialized successfully")


# Default configuration instance
DEFAULT_CONFIG = Config()

# Quick mode configuration (for testing)
QUICK_CONFIG = Config(
    data=DataConfig(
        year=2024,
        race_names=["Bahrain", "Saudi Arabia"],
        quick_mode=True,
    ),
    model=ModelConfig(
        xgb_n_estimators=50,  # Faster training
    ),
)
