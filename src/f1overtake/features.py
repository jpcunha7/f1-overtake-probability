"""Feature engineering for overtake prediction.

Author: João Pedro Cunha

FEATURE CATEGORIES:
1. Gap features: Time gap to car ahead
2. Pace features: Relative pace indicators (recent lap times)
3. Tire features: Tire age, compound differences
4. Track features: Track-specific encoding
5. Race phase features: Lap number, race progress
6. Temporal features: Lagged gaps, closing rates, rolling pace deltas
"""

import logging

import numpy as np
import pandas as pd

from f1overtake.config import FeatureConfig

logger = logging.getLogger(__name__)


def engineer_features(
    opportunities: pd.DataFrame, lap_data: pd.DataFrame, config: FeatureConfig = FeatureConfig()
) -> pd.DataFrame:
    """Engineer features for overtake prediction.

    Args:
        opportunities: DataFrame with overtake opportunities
        lap_data: Full lap data
        config: Feature configuration

    Returns:
        DataFrame with engineered features
    """
    logger.info("Engineering features...")

    df = opportunities.copy()

    # 1. Gap features (already have Gap)
    df = _add_gap_features(df)

    # 2. Pace features
    df = _add_pace_features(df, lap_data, config.pace_window)

    # 3. Tire features
    df = _add_tire_features(df, lap_data)

    # 4. Track features
    df = _add_track_features(df)

    # 5. Race phase features
    df = _add_race_phase_features(df)

    # 6. Temporal features
    if config.enable_temporal_features:
        df = _add_temporal_features(df, lap_data, config)

    logger.info(f"Engineered {len(df.columns)} features")

    return df


def _add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add gap-related features."""
    # Gap already exists from labeling
    # Add categorical gap bins
    df["GapBin"] = pd.cut(
        df["Gap"],
        bins=[0, 0.5, 1.0, 2.0, 3.0, np.inf],
        labels=["very_close", "close", "medium", "far", "very_far"],
    )

    return df


def _add_pace_features(
    df: pd.DataFrame, lap_data: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Add pace-related features.

    Calculates rolling average lap times and relative pace.
    """
    # Create a helper function to get recent pace
    def get_recent_pace(race_name, driver, lap_num, window):
        """Get average lap time over recent laps."""
        driver_laps = lap_data[
            (lap_data["RaceName"] == race_name)
            & (lap_data["Driver"] == driver)
            & (lap_data["LapNumber"] >= lap_num - window)
            & (lap_data["LapNumber"] < lap_num)
        ]

        if len(driver_laps) == 0:
            return None

        return driver_laps["LapTimeSeconds"].mean()

    # Calculate recent pace for driver and driver ahead
    df["DriverRecentPace"] = df.apply(
        lambda row: get_recent_pace(row["RaceName"], row["Driver"], row["LapNumber"], window),
        axis=1,
    )

    df["AheadRecentPace"] = df.apply(
        lambda row: get_recent_pace(
            row["RaceName"], row["DriverAhead"], row["LapNumber"], window
        ),
        axis=1,
    )

    # Relative pace (positive means driver is faster)
    df["RelativePace"] = df["AheadRecentPace"] - df["DriverRecentPace"]

    # Pace ratio
    df["PaceRatio"] = df["DriverRecentPace"] / df["AheadRecentPace"]

    return df


def _add_tire_features(df: pd.DataFrame, lap_data: pd.DataFrame) -> pd.DataFrame:
    """Add tire-related features."""

    # Get tire info for driver ahead
    def get_ahead_tire_info(race_name, driver_ahead, lap_num):
        """Get tire compound and age for driver ahead."""
        ahead_lap = lap_data[
            (lap_data["RaceName"] == race_name)
            & (lap_data["Driver"] == driver_ahead)
            & (lap_data["LapNumber"] == lap_num)
        ]

        if len(ahead_lap) == 0:
            return None, None

        return ahead_lap.iloc[0].get("Compound"), ahead_lap.iloc[0].get("TyreLife")

    # Extract ahead tire info
    ahead_tire_info = df.apply(
        lambda row: get_ahead_tire_info(row["RaceName"], row["DriverAhead"], row["LapNumber"]),
        axis=1,
    )

    df["AheadCompound"] = [x[0] for x in ahead_tire_info]
    df["AheadTyreLife"] = [x[1] for x in ahead_tire_info]

    # Tire age difference (positive means driver has fresher tires)
    df["TireAgeDiff"] = df["AheadTyreLife"] - df["TyreLife"]

    # Same compound flag
    df["SameCompound"] = (df["Compound"] == df["AheadCompound"]).astype(int)

    # Compound encoding (ordinal: SOFT=2, MEDIUM=1, HARD=0)
    compound_map = {"SOFT": 2, "MEDIUM": 1, "HARD": 0}
    df["CompoundNumeric"] = df["Compound"].map(compound_map)
    df["AheadCompoundNumeric"] = df["AheadCompound"].map(compound_map)

    # Compound advantage (positive means driver has softer tire)
    df["CompoundAdvantage"] = df["CompoundNumeric"] - df["AheadCompoundNumeric"]

    return df


def _add_track_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add track-related features."""
    # Simple track encoding (can be expanded with track characteristics)
    df["TrackEncoded"] = pd.factorize(df["RaceName"])[0]

    return df


def _add_race_phase_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add race phase features."""

    # Lap number normalized by total race laps (approximation)
    # Most F1 races are 50-70 laps
    df["RaceProgress"] = df["LapNumber"] / 60.0  # Normalize to approximate race length

    # Race phase categorical
    def get_race_phase(lap_num):
        if lap_num <= 15:
            return "early"
        elif lap_num <= 40:
            return "middle"
        else:
            return "late"

    df["RacePhase"] = df["LapNumber"].apply(get_race_phase)

    return df


def _add_temporal_features(
    df: pd.DataFrame, lap_data: pd.DataFrame, config: FeatureConfig
) -> pd.DataFrame:
    """Add temporal features (lagged gaps, closing rates, rolling pace deltas).

    Args:
        df: DataFrame with opportunities
        lap_data: Full lap data
        config: Feature configuration

    Returns:
        DataFrame with added temporal features
    """
    logger.info("Adding temporal features...")

    # Create a helper to get historical gap values
    def get_lagged_gap(race_name, driver, driver_ahead, lap_num, lag):
        """Get gap from L-lag laps ago."""
        target_lap = lap_num - lag
        if target_lap < 1:
            return None

        # Find the same driver pair at the earlier lap
        historical = df[
            (df["RaceName"] == race_name)
            & (df["Driver"] == driver)
            & (df["DriverAhead"] == driver_ahead)
            & (df["LapNumber"] == target_lap)
        ]

        if len(historical) > 0:
            return historical.iloc[0]["Gap"]
        return None

    # Add lagged gap features
    for lag in config.lagged_gap_laps:
        col_name = f"Gap_L{lag}"
        df[col_name] = df.apply(
            lambda row: get_lagged_gap(
                row["RaceName"], row["Driver"], row["DriverAhead"], row["LapNumber"], lag
            ),
            axis=1,
        )
        logger.info(f"Added {col_name}: {df[col_name].notna().sum()} non-null values")

    # Calculate closing rate (gap change over recent laps)
    if len(config.lagged_gap_laps) > 0:
        max_lag = max(config.lagged_gap_laps)
        lag_col = f"Gap_L{max_lag}"

        if lag_col in df.columns:
            # Closing rate: positive means gap is closing (getting faster relative to car ahead)
            df["ClosingRate"] = (df[lag_col] - df["Gap"]) / max_lag
            logger.info(f"Added ClosingRate: {df['ClosingRate'].notna().sum()} non-null values")

    # Add rolling pace delta (change in relative pace over window)
    def get_rolling_pace_delta(race_name, driver, driver_ahead, lap_num, window):
        """Calculate change in relative pace over window."""
        # Get relative pace from window laps ago
        target_lap = lap_num - window
        if target_lap < 1:
            return None

        historical = df[
            (df["RaceName"] == race_name)
            & (df["Driver"] == driver)
            & (df["DriverAhead"] == driver_ahead)
            & (df["LapNumber"] == target_lap)
        ]

        if len(historical) > 0 and "RelativePace" in historical.columns:
            past_pace = historical.iloc[0]["RelativePace"]
            current_pace = df[
                (df["RaceName"] == race_name)
                & (df["Driver"] == driver)
                & (df["DriverAhead"] == driver_ahead)
                & (df["LapNumber"] == lap_num)
            ]["RelativePace"].iloc[0] if len(df[
                (df["RaceName"] == race_name)
                & (df["Driver"] == driver)
                & (df["DriverAhead"] == driver_ahead)
                & (df["LapNumber"] == lap_num)
            ]) > 0 else None

            if past_pace is not None and current_pace is not None:
                return current_pace - past_pace
        return None

    df["RollingPaceDelta"] = df.apply(
        lambda row: get_rolling_pace_delta(
            row["RaceName"],
            row["Driver"],
            row["DriverAhead"],
            row["LapNumber"],
            config.closing_rate_window
        ),
        axis=1,
    )
    logger.info(f"Added RollingPaceDelta: {df['RollingPaceDelta'].notna().sum()} non-null values")

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select final feature set for modeling.

    Args:
        df: DataFrame with all engineered features

    Returns:
        DataFrame with selected features and target
    """
    # Core numerical features
    numerical_features = [
        "Gap",
        "RelativePace",
        "PaceRatio",
        "TyreLife",
        "AheadTyreLife",
        "TireAgeDiff",
        "CompoundAdvantage",
        "RaceProgress",
        "Position",
    ]

    # Temporal features (added if enabled)
    temporal_features = [
        "Gap_L1",
        "Gap_L2",
        "ClosingRate",
        "RollingPaceDelta",
    ]

    # Categorical features (need encoding)
    categorical_features = [
        "GapBin",
        "SameCompound",
        "RacePhase",
        "TrackEncoded",
    ]

    # Target
    target = ["Overtake"]

    # Metadata (not used for modeling)
    metadata = ["RaceName", "LapNumber", "Driver", "DriverAhead"]

    # Combine and filter to available columns
    feature_cols = numerical_features + temporal_features + categorical_features
    available_features = [col for col in feature_cols if col in df.columns]

    all_cols = metadata + available_features + target
    available_cols = [col for col in all_cols if col in df.columns]

    result = df[available_cols].copy()

    # Drop rows with missing critical features
    result = result.dropna(subset=available_features)

    logger.info(
        f"Selected {len(available_features)} features, " f"{len(result)} samples after dropping NaN"
    )

    return result
