"""Build overtake prediction dataset from raw F1 data.

Author: João Pedro Cunha
"""

import logging
from pathlib import Path

import pandas as pd

from f1overtake.config import Config, DEFAULT_CONFIG
from f1overtake.data_loader import load_multiple_races
from f1overtake.features import engineer_features, select_features
from f1overtake.labels import create_overtake_labels

logger = logging.getLogger(__name__)


def build_dataset(config: Config = DEFAULT_CONFIG, force_rebuild: bool = False) -> pd.DataFrame:
    """Build complete dataset with features and labels.

    Args:
        config: Configuration object
        force_rebuild: If True, rebuild even if cached dataset exists

    Returns:
        DataFrame with features and labels
    """
    # Use different cache files for different configurations
    num_races = len(config.data.race_names)
    dataset_path = Path(config.data.dataset_dir) / f"overtake_dataset_{num_races}_races.parquet"

    # Check for cached dataset
    if dataset_path.exists() and not force_rebuild:
        logger.info(f"Loading cached dataset from {dataset_path}")
        return pd.read_parquet(dataset_path)

    logger.info("Building dataset from scratch...")

    # Step 1: Load raw lap data
    logger.info("Step 1/4: Loading race data")
    lap_data = load_multiple_races(
        year=config.data.year,
        race_names=config.data.race_names,
        cache_dir=config.data.cache_dir,
    )

    # Step 2: Create labels
    logger.info("Step 2/4: Creating overtake labels")
    opportunities = create_overtake_labels(lap_data, config.labels)

    if len(opportunities) == 0:
        raise ValueError("No overtake opportunities generated! Check labeling configuration.")

    # Step 3: Engineer features
    logger.info("Step 3/4: Engineering features")
    featured_data = engineer_features(opportunities, lap_data, config.features)

    # Step 4: Select final features
    logger.info("Step 4/4: Selecting final features")
    dataset = select_features(featured_data)

    # Save to cache
    logger.info(f"Saving dataset to {dataset_path}")
    dataset.to_parquet(dataset_path, index=False)

    logger.info(
        f"Dataset built: {len(dataset)} samples, "
        f"{dataset['Overtake'].sum()} positive labels "
        f"({dataset['Overtake'].mean()*100:.1f}% overtake rate)"
    )

    return dataset


def get_dataset_summary(dataset: pd.DataFrame) -> dict:
    """Get summary statistics of dataset.

    Args:
        dataset: Dataset DataFrame

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "total_samples": len(dataset),
        "positive_samples": int(dataset["Overtake"].sum()),
        "negative_samples": int((~dataset["Overtake"]).sum()),
        "overtake_rate": float(dataset["Overtake"].mean()),
        "num_races": dataset["RaceName"].nunique(),
        "num_drivers": dataset["Driver"].nunique(),
        "features": [
            col for col in dataset.columns if col not in ["Overtake", "RaceName", "Driver"]
        ],
    }

    return summary


if __name__ == "__main__":
    """Command-line interface for building dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Build overtake prediction dataset")
    parser.add_argument(
        "--quick", action="store_true", help="Use quick mode (fewer races)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild even if cached"
    )

    args = parser.parse_args()

    # Select config
    from f1overtake.config import QUICK_CONFIG

    cfg = QUICK_CONFIG if args.quick else DEFAULT_CONFIG

    # Build dataset
    dataset = build_dataset(cfg, force_rebuild=args.force)

    # Print summary
    summary = get_dataset_summary(dataset)
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Positive labels (overtakes): {summary['positive_samples']}")
    print(f"Negative labels: {summary['negative_samples']}")
    print(f"Overtake rate: {summary['overtake_rate']*100:.2f}%")
    print(f"Number of races: {summary['num_races']}")
    print(f"Number of drivers: {summary['num_drivers']}")
    print(f"Number of features: {len(summary['features'])}")
    print("=" * 60)
