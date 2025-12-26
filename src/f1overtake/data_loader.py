"""Data loading utilities for F1 race data using FastF1.

Author: João Pedro Cunha
"""

import logging
from pathlib import Path
from typing import Dict, List

import fastf1
import pandas as pd


logger = logging.getLogger(__name__)


def enable_cache(cache_dir: str = "cache") -> None:
    """Enable FastF1 caching to speed up data loading."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))
    logger.info(f"FastF1 cache enabled at: {cache_path}")


def load_race_session(year: int, race_name: str):
    """Load a race session using FastF1.

    Args:
        year: Season year
        race_name: Name of the race (e.g., "Bahrain")

    Returns:
        FastF1 Session object
    """
    logger.info(f"Loading race session: {year} {race_name}")
    session = fastf1.get_session(year, race_name, "R")
    session.load()
    logger.info(f"Loaded session: {session.event['EventName']}")
    return session


def extract_lap_data(session) -> pd.DataFrame:
    """Extract lap-by-lap data from a session.

    Args:
        session: FastF1 session object

    Returns:
        DataFrame with columns: LapNumber, Driver, Position, LapTime, Compound,
                                TyreLife, Stint, IsPersonalBest, etc.
    """
    laps = session.laps

    # Select relevant columns
    columns = [
        "LapNumber",
        "Driver",
        "Position",
        "LapTime",
        "Compound",
        "TyreLife",
        "Stint",
        "IsPersonalBest",
        "Team",
        "PitInTime",
        "PitOutTime",
    ]

    # Filter to available columns
    available_cols = [col for col in columns if col in laps.columns]
    lap_data = laps[available_cols].copy()

    # Convert LapTime to seconds
    lap_data["LapTimeSeconds"] = lap_data["LapTime"].dt.total_seconds()

    # Add race metadata
    lap_data["RaceName"] = session.event["EventName"]
    lap_data["Year"] = session.event["EventDate"].year
    lap_data["TrackId"] = session.event["EventName"]  # Simplified track identifier

    # Create pit flags
    lap_data["IsPitLap"] = (~lap_data["PitInTime"].isna()) | (~lap_data["PitOutTime"].isna())

    return lap_data


def load_multiple_races(
    year: int, race_names: List[str], cache_dir: str = "cache"
) -> pd.DataFrame:
    """Load multiple races and combine into single DataFrame.

    Args:
        year: Season year
        race_names: List of race names
        cache_dir: Cache directory for FastF1

    Returns:
        Combined DataFrame with all race data
    """
    enable_cache(cache_dir)

    all_laps = []

    for race_name in race_names:
        try:
            session = load_race_session(year, race_name)
            lap_data = extract_lap_data(session)
            all_laps.append(lap_data)
            logger.info(f"Extracted {len(lap_data)} laps from {race_name}")
        except Exception as e:
            logger.error(f"Failed to load {race_name}: {e}")
            continue

    if not all_laps:
        raise ValueError("No race data could be loaded")

    combined = pd.concat(all_laps, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} laps across {len(all_laps)} races")

    return combined


def get_driver_positions_by_lap(lap_data: pd.DataFrame) -> Dict[int, Dict[str, int]]:
    """Extract position of each driver at each lap.

    Args:
        lap_data: DataFrame with lap data

    Returns:
        Dict[lap_number -> Dict[driver_code -> position]]
    """
    positions = {}

    for lap_num in lap_data["LapNumber"].unique():
        lap_positions = lap_data[lap_data["LapNumber"] == lap_num][
            ["Driver", "Position"]
        ].dropna()

        positions[lap_num] = dict(zip(lap_positions["Driver"], lap_positions["Position"]))

    return positions
