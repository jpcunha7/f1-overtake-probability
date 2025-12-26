"""Expected Overtakes (xO) metric calculation and analysis.

Author: João Pedro Cunha

The xO (Expected Overtakes) metric represents the sum of predicted overtake
probabilities for a driver across a race. It provides an expected value for
the number of overtakes a driver should make given the opportunities they had.

IMPORTANT: This metric relies on well-calibrated probability estimates.
Poorly calibrated models will produce misleading xO values.

Example:
    If a driver has 10 overtake opportunities with probabilities:
    [0.1, 0.2, 0.3, 0.05, 0.15, 0.4, 0.1, 0.25, 0.2, 0.1]
    xO = sum of probabilities = 1.85

This suggests the driver was expected to make approximately 1.85 overtakes
based on their opportunities. Comparing xO to actual overtakes can reveal
drivers who over/underperformed expectations.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from f1overtake.config import XOConfig
from f1overtake.split import prepare_xy

logger = logging.getLogger(__name__)


def calculate_xo(
    df: pd.DataFrame,
    model,
    config: XOConfig = XOConfig(),
) -> pd.DataFrame:
    """Calculate xO (Expected Overtakes) for each driver in each race.

    Args:
        df: DataFrame with overtake opportunities and features
        model: Trained model with predict_proba method
        config: xO configuration

    Returns:
        DataFrame with columns: RaceName, Driver, xO, ActualOvertakes, Opportunities
    """
    logger.info("Calculating xO (Expected Overtakes)...")

    # Prepare features
    X, y = prepare_xy(df)

    # Align features if needed
    if hasattr(model, "feature_names_"):
        X = X[model.feature_names_]

    # Get predictions
    try:
        probabilities = model.predict_proba(X)[:, 1]
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        return pd.DataFrame()

    # Add predictions to dataframe
    results_df = df[["RaceName", "Driver", "Overtake"]].copy()
    results_df["Probability"] = probabilities

    # Calculate xO per driver per race
    xo_results = []

    for (race_name, driver), group in results_df.groupby(["RaceName", "Driver"]):
        n_opportunities = len(group)

        # Skip if too few opportunities
        if n_opportunities < config.min_opportunities:
            continue

        xo = group["Probability"].sum()
        actual = group["Overtake"].sum()

        xo_results.append({
            "RaceName": race_name,
            "Driver": driver,
            "xO": xo,
            "ActualOvertakes": int(actual),
            "Opportunities": n_opportunities,
            "xO_per_Opportunity": xo / n_opportunities if n_opportunities > 0 else 0,
            "Actual_per_Opportunity": actual / n_opportunities if n_opportunities > 0 else 0,
            "Delta": actual - xo,  # Positive means overperformed
        })

    xo_df = pd.DataFrame(xo_results)

    logger.info(f"Calculated xO for {len(xo_df)} driver-race combinations")

    return xo_df


def create_xo_leaderboard(
    xo_df: pd.DataFrame,
    race_name: Optional[str] = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Create xO leaderboard showing drivers who over/underperformed.

    Args:
        xo_df: DataFrame from calculate_xo
        race_name: Optional race name to filter by (None = aggregate all races)
        top_n: Number of top drivers to show

    Returns:
        Leaderboard DataFrame sorted by Delta (actual - xO)
    """
    if race_name:
        filtered = xo_df[xo_df["RaceName"] == race_name].copy()
    else:
        # Aggregate across all races
        filtered = xo_df.groupby("Driver").agg({
            "xO": "sum",
            "ActualOvertakes": "sum",
            "Opportunities": "sum",
            "Delta": "sum",
        }).reset_index()

        # Recalculate per-opportunity metrics
        filtered["xO_per_Opportunity"] = filtered["xO"] / filtered["Opportunities"]
        filtered["Actual_per_Opportunity"] = filtered["ActualOvertakes"] / filtered["Opportunities"]

    # Sort by Delta (biggest overperformers first)
    leaderboard = filtered.sort_values("Delta", ascending=False).head(top_n)

    return leaderboard


def visualize_xo_analysis(
    xo_df: pd.DataFrame,
    race_name: Optional[str] = None,
    config: XOConfig = XOConfig(),
) -> go.Figure:
    """Create visualization of xO analysis.

    Args:
        xo_df: DataFrame from calculate_xo
        race_name: Optional race name to filter by
        config: xO configuration

    Returns:
        Plotly figure with xO analysis
    """
    if race_name:
        data = xo_df[xo_df["RaceName"] == race_name].copy()
        title = f"xO Analysis - {race_name}"
    else:
        # Aggregate by driver
        data = xo_df.groupby("Driver").agg({
            "xO": "sum",
            "ActualOvertakes": "sum",
            "Opportunities": "sum",
            "Delta": "sum",
        }).reset_index()
        title = "xO Analysis - All Races"

    # Sort by Delta
    data = data.sort_values("Delta", ascending=True)

    # Create figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Expected vs Actual Overtakes", "Performance Delta"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]],
        horizontal_spacing=0.15,
    )

    # Left plot: Expected vs Actual scatter
    fig.add_trace(
        go.Scatter(
            x=data["xO"],
            y=data["ActualOvertakes"],
            mode="markers+text",
            text=data["Driver"],
            textposition="top center",
            marker=dict(
                size=10,
                color=data["Delta"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Delta", x=0.45),
            ),
            name="Drivers",
        ),
        row=1, col=1,
    )

    # Add diagonal line (perfect calibration)
    max_val = max(data["xO"].max(), data["ActualOvertakes"].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Perfect Calibration",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Right plot: Delta bar chart
    colors = ["green" if d > 0 else "red" for d in data["Delta"]]
    fig.add_trace(
        go.Bar(
            x=data["Delta"],
            y=data["Driver"],
            orientation="h",
            marker=dict(color=colors),
            name="Delta",
        ),
        row=1, col=2,
    )

    # Update layout
    fig.update_xaxes(title_text="Expected Overtakes (xO)", row=1, col=1)
    fig.update_yaxes(title_text="Actual Overtakes", row=1, col=1)
    fig.update_xaxes(title_text="Delta (Actual - xO)", row=1, col=2)
    fig.update_yaxes(title_text="Driver", row=1, col=2)

    fig.update_layout(
        title_text=title,
        height=600,
        showlegend=True,
    )

    # Add calibration warning if enabled
    if config.show_calibration_warning:
        fig.add_annotation(
            text="Note: xO accuracy depends on model calibration quality",
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.1,
            showarrow=False,
            font=dict(size=10, color="gray"),
            xanchor="center",
        )

    return fig


def get_race_xo_summary(
    df: pd.DataFrame,
    model,
    race_name: str,
    config: XOConfig = XOConfig(),
) -> Tuple[pd.DataFrame, Dict]:
    """Get detailed xO summary for a specific race.

    Args:
        df: DataFrame with opportunities and features
        model: Trained model
        race_name: Race name
        config: xO configuration

    Returns:
        Tuple of (driver leaderboard DataFrame, race statistics dict)
    """
    # Filter to race
    race_df = df[df["RaceName"] == race_name].copy()

    if len(race_df) == 0:
        logger.warning(f"No data found for race: {race_name}")
        return pd.DataFrame(), {}

    # Calculate xO
    xo_df = calculate_xo(race_df, model, config)

    # Create leaderboard
    leaderboard = create_xo_leaderboard(xo_df, race_name=race_name, top_n=25)

    # Calculate race statistics
    stats = {
        "total_opportunities": len(race_df),
        "total_actual_overtakes": int(race_df["Overtake"].sum()),
        "total_expected_overtakes": xo_df["xO"].sum(),
        "n_drivers": xo_df["Driver"].nunique(),
        "avg_xo_per_driver": xo_df["xO"].mean(),
        "avg_actual_per_driver": xo_df["ActualOvertakes"].mean(),
    }

    logger.info(f"Race summary for {race_name}:")
    logger.info(f"  Total opportunities: {stats['total_opportunities']}")
    logger.info(f"  Actual overtakes: {stats['total_actual_overtakes']}")
    logger.info(f"  Expected overtakes (xO): {stats['total_expected_overtakes']:.2f}")

    return leaderboard, stats


def export_xo_table(
    xo_df: pd.DataFrame,
    output_path: str,
    format: str = "csv",
) -> None:
    """Export xO results to file.

    Args:
        xo_df: DataFrame from calculate_xo
        output_path: Output file path
        format: Output format ('csv', 'excel', 'html')
    """
    if format == "csv":
        xo_df.to_csv(output_path, index=False)
    elif format == "excel":
        xo_df.to_excel(output_path, index=False)
    elif format == "html":
        xo_df.to_html(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Exported xO table to {output_path}")
