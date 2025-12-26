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

    # Exclude safety car laps
    exclude_safety_car: bool = True
    safety_car_threshold_pct: float = 0.15  # % of drivers with slow laps

    # Minimum stint length to consider
    min_stint_length: int = 3


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    # Rolling window sizes for pace calculations
    pace_window: int = 3

    # Temporal features
    enable_temporal_features: bool = True
    lagged_gap_laps: List[int] = field(default_factory=lambda: [1, 2])  # Look back 1 and 2 laps
    closing_rate_window: int = 3  # Window for calculating closing rate

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
    xgb_min_child_weight: int = 1
    xgb_gamma: float = 0.0
    xgb_colsample_bytree: float = 1.0

    # Class imbalance handling
    use_class_weights: bool = True

    # Optuna hyperparameter tuning
    use_optuna: bool = False
    optuna_n_trials: int = 50
    optuna_cv_folds: int = 5
    optuna_timeout: int = 3600  # 1 hour timeout

    # Calibration method
    calibration_method: str = "isotonic"  # or 'sigmoid'

    # Test split
    test_size: float = 0.3  # 30% of races for testing

    # Performance targets
    target_roc_auc: float = 0.75
    target_pr_auc_multiplier: float = 2.0  # 2x baseline


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Threshold for binary classification
    decision_threshold: float = 0.5

    # Generate calibration curves
    n_bins: int = 10

    # Model report generation
    generate_html_report: bool = True
    report_n_examples: int = 10  # Number of example predictions to show


@dataclass
class XOConfig:
    """Expected Overtakes (xO) metric configuration."""

    # Minimum number of overtake opportunities to calculate xO
    min_opportunities: int = 5

    # Include calibration caveat in outputs
    show_calibration_warning: bool = True


@dataclass
class Config:
    """Master configuration object."""

    data: DataConfig = field(default_factory=DataConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    xo: XOConfig = field(default_factory=XOConfig)

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
