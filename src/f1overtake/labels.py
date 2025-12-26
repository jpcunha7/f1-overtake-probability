"""Overtake labeling logic with pit stop filtering.

Author: João Pedro Cunha

LABELING STRATEGY:
An overtake opportunity is defined as follows:
- For each lap L and each driver i:
  1. Identify the car directly ahead (driver j at position p-1)
  2. Check if driver i overtakes driver j within the next K laps
  3. Label y=1 if overtake occurs, y=0 otherwise

PIT CONFOUNDING MITIGATION:
- Exclude laps where either driver pits in current lap or lookahead window
- Exclude laps with compound changes
- Exclude outlier lap times (formation laps, safety cars)
- Require minimum stint length

LIMITATIONS:
- Position changes may occur without on-track overtakes (pit strategy)
- Label noise exists due to incomplete pit detection
- Safety car periods may not be fully filtered
- DRS and track-specific factors not explicitly modeled
"""

import logging
from typing import List

import pandas as pd

from f1overtake.config import LabelConfig

logger = logging.getLogger(__name__)


def create_overtake_labels(
    lap_data: pd.DataFrame, config: LabelConfig = LabelConfig()
) -> pd.DataFrame:
    """Create overtake labels for each lap and driver pair.

    Args:
        lap_data: DataFrame with lap data (must have Position, Driver, LapNumber)
        config: Label configuration

    Returns:
        DataFrame with overtake opportunities and labels
    """
    logger.info("Creating overtake labels...")

    # Handle empty input
    if len(lap_data) == 0:
        logger.warning("Empty lap data provided")
        return pd.DataFrame()

    # Sort by race, lap, position
    lap_data = lap_data.sort_values(["RaceName", "LapNumber", "Position"]).copy()

    opportunities = []

    # Group by race to handle each race separately
    for race_name, race_df in lap_data.groupby("RaceName"):
        logger.info(f"Processing race: {race_name}")

        # Get unique lap numbers
        lap_numbers = sorted(race_df["LapNumber"].unique())

        for lap_num in lap_numbers:
            # Skip if we can't look ahead
            if lap_num + config.lookahead_laps not in lap_numbers:
                continue

            # Get current lap data
            current_lap = race_df[race_df["LapNumber"] == lap_num]

            # Get lookahead lap data
            future_lap = race_df[race_df["LapNumber"] == lap_num + config.lookahead_laps]

            # For each driver, find the car ahead and check for overtake
            for _, row in current_lap.iterrows():
                driver = row["Driver"]
                position = row["Position"]

                # Skip if position is missing or first
                if pd.isna(position) or position <= 1:
                    continue

                # Find driver directly ahead (position - 1)
                ahead_position = position - 1
                ahead_drivers = current_lap[current_lap["Position"] == ahead_position]

                if len(ahead_drivers) == 0:
                    continue

                ahead_driver = ahead_drivers.iloc[0]["Driver"]

                # Check exclusion criteria
                if config.exclude_pit_laps:
                    # Check if either driver pits in current or lookahead window
                    if _check_pit_activity(
                        race_df, driver, lap_num, config.lookahead_laps
                    ) or _check_pit_activity(race_df, ahead_driver, lap_num, config.lookahead_laps):
                        continue

                # Check for outlier lap times
                if config.exclude_outliers:
                    if _is_outlier_lap(row, race_df, config.outlier_threshold):
                        continue

                # Check for safety car laps
                if config.exclude_safety_car:
                    if _is_safety_car_lap(race_df, lap_num, config.safety_car_threshold_pct):
                        continue

                # Calculate gap if available
                gap = _calculate_gap(current_lap, driver, ahead_driver)

                # Skip if gap is too large
                if gap is not None and gap > config.max_gap:
                    continue

                # Determine if overtake occurred
                overtake = _check_overtake(driver, ahead_driver, future_lap)

                # Create opportunity record
                opportunity = {
                    "RaceName": race_name,
                    "LapNumber": lap_num,
                    "Driver": driver,
                    "DriverAhead": ahead_driver,
                    "Position": position,
                    "PositionAhead": ahead_position,
                    "Gap": gap,
                    "Overtake": overtake,
                    "LapTimeSeconds": row.get("LapTimeSeconds"),
                    "Compound": row.get("Compound"),
                    "TyreLife": row.get("TyreLife"),
                }

                opportunities.append(opportunity)

    if not opportunities:
        logger.warning("No overtake opportunities found!")
        return pd.DataFrame()

    result = pd.DataFrame(opportunities)
    logger.info(
        f"Created {len(result)} overtake opportunities, "
        f"{result['Overtake'].sum()} positive labels "
        f"({result['Overtake'].mean()*100:.1f}% overtake rate)"
    )

    return result


def _check_pit_activity(race_df: pd.DataFrame, driver: str, start_lap: int, lookahead: int) -> bool:
    """Check if driver pits within a lap window.

    Improved pit detection using multiple signals:
    - IsPitLap column if available
    - Compound changes
    - Large lap time spikes (pit stops add 20-30 seconds)
    - TyreLife resets

    Args:
        race_df: Race DataFrame
        driver: Driver code
        start_lap: Starting lap number
        lookahead: Number of laps to check ahead

    Returns:
        True if pit activity detected
    """
    driver_laps = race_df[
        (race_df["Driver"] == driver)
        & (race_df["LapNumber"] >= start_lap)
        & (race_df["LapNumber"] <= start_lap + lookahead)
    ].sort_values("LapNumber")

    if len(driver_laps) == 0:
        return False

    # Check for pit stops
    if "IsPitLap" in driver_laps.columns:
        if driver_laps["IsPitLap"].any():
            return True

    # Check for compound changes (indicator of pit stop)
    if "Compound" in driver_laps.columns:
        compounds = driver_laps["Compound"].dropna().unique()
        if len(compounds) > 1:
            return True

    # Check for TyreLife resets (drops significantly)
    if "TyreLife" in driver_laps.columns:
        tyre_lives = driver_laps["TyreLife"].dropna()
        if len(tyre_lives) > 1:
            # If tyre life decreases by more than 5 laps, likely a pit stop
            tyre_diff = tyre_lives.diff()
            if (tyre_diff < -5).any():
                return True

    # Check for abnormally slow laps (pit stops add 20-30 seconds)
    if "LapTimeSeconds" in driver_laps.columns:
        lap_times = driver_laps["LapTimeSeconds"].dropna()
        if len(lap_times) > 0:
            # Get median lap time for this driver in this race
            all_driver_laps = race_df[race_df["Driver"] == driver]["LapTimeSeconds"].dropna()
            if len(all_driver_laps) > 5:
                median_time = all_driver_laps.median()
                # If any lap is more than 20 seconds slower than median, likely pit stop
                if (lap_times > median_time + 20).any():
                    return True

    return False


def _is_outlier_lap(row: pd.Series, race_df: pd.DataFrame, threshold: float) -> bool:
    """Check if lap time is an outlier.

    Args:
        row: Lap row
        race_df: Full race DataFrame
        threshold: Number of standard deviations

    Returns:
        True if outlier
    """
    if "LapTimeSeconds" not in row or pd.isna(row["LapTimeSeconds"]):
        return True

    lap_time = row["LapTimeSeconds"]

    # Get all lap times for this race
    race_times = race_df["LapTimeSeconds"].dropna()

    if len(race_times) < 10:
        return False

    median_time = race_times.median()
    std_time = race_times.std()

    # Check if lap time is too far from median
    z_score = abs(lap_time - median_time) / std_time if std_time > 0 else 0

    return z_score > threshold


def _calculate_gap(current_lap: pd.DataFrame, driver: str, ahead_driver: str) -> float:
    """Calculate time gap between two drivers.

    Args:
        current_lap: Current lap data
        driver: Driver code
        ahead_driver: Driver ahead code

    Returns:
        Gap in seconds (None if unavailable)
    """
    driver_row = current_lap[current_lap["Driver"] == driver]
    ahead_row = current_lap[current_lap["Driver"] == ahead_driver]

    if len(driver_row) == 0 or len(ahead_row) == 0:
        return None

    driver_time = driver_row.iloc[0].get("LapTimeSeconds")
    ahead_time = ahead_row.iloc[0].get("LapTimeSeconds")

    if pd.isna(driver_time) or pd.isna(ahead_time):
        return None

    # Gap is difference in lap times (approximation)
    return abs(driver_time - ahead_time)


def _is_safety_car_lap(race_df: pd.DataFrame, lap_num: int, threshold_pct: float = 0.15) -> bool:
    """Detect if a lap is under safety car using lap time anomalies.

    During safety car periods, all drivers slow down significantly.
    We detect this by checking if most drivers have abnormally slow lap times.

    Args:
        race_df: Race DataFrame
        lap_num: Lap number to check
        threshold_pct: Percentage of drivers that must have slow laps

    Returns:
        True if likely safety car lap
    """
    lap_data = race_df[race_df["LapNumber"] == lap_num]

    if len(lap_data) < 5:
        return False

    # Get lap times for this lap
    lap_times = lap_data["LapTimeSeconds"].dropna()

    if len(lap_times) < 5:
        return False

    # Get typical lap times for this race (median of all laps)
    all_lap_times = race_df["LapTimeSeconds"].dropna()

    if len(all_lap_times) < 50:
        return False

    median_time = all_lap_times.median()

    # Check how many drivers are significantly slower (>10 seconds)
    slow_count = (lap_times > median_time + 10).sum()
    slow_pct = slow_count / len(lap_times)

    # If more than threshold_pct of drivers are slow, likely safety car
    return slow_pct > threshold_pct


def label_sensitivity_analysis(
    lap_data: pd.DataFrame,
    lookahead_values: List[int] = [1, 2, 3],
    config: LabelConfig = LabelConfig(),
) -> pd.DataFrame:
    """Analyze how labels change with different lookahead values.

    This helps understand label sensitivity and potential noise.

    Args:
        lap_data: Full lap data
        lookahead_values: List of lookahead values to test
        config: Label configuration

    Returns:
        DataFrame with label statistics for each lookahead value
    """
    logger.info("Running label sensitivity analysis...")

    results = []

    for lookahead in lookahead_values:
        config_copy = LabelConfig(
            lookahead_laps=lookahead,
            max_gap=config.max_gap,
            exclude_pit_laps=config.exclude_pit_laps,
            exclude_outliers=config.exclude_outliers,
        )

        opportunities = create_overtake_labels(lap_data, config_copy)

        if len(opportunities) > 0:
            results.append(
                {
                    "lookahead": lookahead,
                    "total_opportunities": len(opportunities),
                    "overtakes": int(opportunities["Overtake"].sum()),
                    "overtake_rate": opportunities["Overtake"].mean(),
                }
            )

    results_df = pd.DataFrame(results)

    logger.info("Label sensitivity analysis complete:")
    logger.info(results_df.to_string())

    return results_df


def _check_overtake(driver: str, ahead_driver: str, future_lap: pd.DataFrame) -> bool:
    """Check if driver overtook the ahead driver.

    Args:
        driver: Driver code
        ahead_driver: Driver who was ahead
        future_lap: Future lap data

    Returns:
        True if overtake occurred
    """
    driver_future = future_lap[future_lap["Driver"] == driver]
    ahead_future = future_lap[future_lap["Driver"] == ahead_driver]

    if len(driver_future) == 0 or len(ahead_future) == 0:
        return False

    driver_pos = driver_future.iloc[0]["Position"]
    ahead_pos = ahead_future.iloc[0]["Position"]

    if pd.isna(driver_pos) or pd.isna(ahead_pos):
        return False

    # Overtake if driver is now ahead
    return driver_pos < ahead_pos
