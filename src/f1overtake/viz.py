"""Visualization utilities for overtake probability model.

Author: João Pedro Cunha
"""

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def create_calibration_plot(calibration_data: dict, model_name: str = "") -> go.Figure:
    """Create calibration curve plot.

    Args:
        calibration_data: Dictionary from get_calibration_data()
        model_name: Name of model for title

    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Calibration Curve", "Prediction Distribution"),
        vertical_spacing=0.15,
    )

    # Calibration curve
    fig.add_trace(
        go.Scatter(
            x=calibration_data["bin_centers"],
            y=calibration_data["bin_means"],
            mode="markers+lines",
            name="Calibration",
            marker=dict(size=10),
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="red", dash="dash", width=2),
        ),
        row=1,
        col=1,
    )

    # Histogram of predictions
    fig.add_trace(
        go.Histogram(
            x=calibration_data["predicted_probs"],
            nbinsx=20,
            name="Predictions",
            marker=dict(color="lightblue"),
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Predicted Probability", row=1, col=1)
    fig.update_yaxes(title_text="Actual Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Probability", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_layout(
        title=f"Model Calibration: {model_name}",
        height=600,
        showlegend=True,
    )

    return fig


def create_feature_importance_plot(importance_df: pd.DataFrame) -> go.Figure:
    """Create feature importance bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns

    Returns:
        Plotly figure
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=importance_df["importance"],
                y=importance_df["feature"],
                orientation="h",
                marker=dict(color="steelblue"),
            )
        ]
    )

    fig.update_layout(
        title="Top Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        yaxis=dict(autorange="reversed"),
    )

    return fig


def create_probability_heatmap(
    model, gap_range: tuple = (0, 3), tire_age_range: tuple = (0, 30)
) -> go.Figure:
    """Create probability heatmap for gap vs tire age.

    Args:
        model: Trained model
        gap_range: Range of gap values
        tire_age_range: Range of tire age values

    Returns:
        Plotly figure
    """
    # Create grid
    gaps = np.linspace(gap_range[0], gap_range[1], 20)
    tire_ages = np.linspace(tire_age_range[0], tire_age_range[1], 20)

    # Create base feature vector (mean values for other features)
    # This is a simplified approach - in practice, you'd want to be more careful
    base_features = {
        "Gap": 1.0,
        "RelativePace": 0.0,
        "PaceRatio": 1.0,
        "TyreLife": 10,
        "AheadTyreLife": 10,
        "TireAgeDiff": 0,
        "CompoundAdvantage": 0,
        "RaceProgress": 0.5,
        "Position": 10,
    }

    # Get feature names from model
    if hasattr(model, "feature_names_"):
        feature_names = model.feature_names_
    else:
        logger.warning("Model does not have feature_names_, using defaults")
        feature_names = list(base_features.keys())

    # Calculate probabilities
    probs = np.zeros((len(tire_ages), len(gaps)))

    for i, tire_age in enumerate(tire_ages):
        for j, gap in enumerate(gaps):
            # Create feature vector
            features = base_features.copy()
            features["Gap"] = gap
            features["TireAgeDiff"] = tire_age  # Simplification

            # Align with model features
            X = pd.DataFrame([features])
            X = X.reindex(columns=feature_names, fill_value=0)

            # Predict
            prob = model.predict_proba(X)[0, 1]
            probs[i, j] = prob

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=probs,
            x=gaps,
            y=tire_ages,
            colorscale="RdYlGn",
            colorbar=dict(title="Overtake Probability"),
        )
    )

    fig.update_layout(
        title="Overtake Probability Heatmap",
        xaxis_title="Gap (seconds)",
        yaxis_title="Tire Age Advantage (laps)",
        height=500,
    )

    return fig


def create_race_timeline_plot(
    race_df: pd.DataFrame, driver: str, model, driver_ahead: str = None
) -> go.Figure:
    """Create timeline of overtake probabilities during a race.

    Args:
        race_df: Race DataFrame with features
        driver: Driver code
        model: Trained model
        driver_ahead: Specific driver ahead to track (optional)

    Returns:
        Plotly figure
    """
    # Filter to driver
    if driver_ahead:
        driver_data = race_df[
            (race_df["Driver"] == driver) & (race_df["DriverAhead"] == driver_ahead)
        ].copy()
    else:
        driver_data = race_df[race_df["Driver"] == driver].copy()

    if len(driver_data) == 0:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this driver",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    # Get predictions
    X, _ = prepare_xy(driver_data)
    if hasattr(model, "feature_names_"):
        X = X[model.feature_names_]

    probs = model.predict_proba(X)[:, 1]
    driver_data["Probability"] = probs

    # Create timeline
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=driver_data["LapNumber"],
            y=driver_data["Probability"],
            mode="lines+markers",
            name="Overtake Probability",
            line=dict(color="blue", width=2),
            marker=dict(size=6),
        )
    )

    # Add threshold line
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold (0.5)"
    )

    # Highlight actual overtakes
    actual_overtakes = driver_data[driver_data["Overtake"] == 1]
    if len(actual_overtakes) > 0:
        fig.add_trace(
            go.Scatter(
                x=actual_overtakes["LapNumber"],
                y=actual_overtakes["Probability"],
                mode="markers",
                name="Actual Overtake",
                marker=dict(size=12, color="green", symbol="star"),
            )
        )

    title = f"Overtake Probability Timeline: {driver}"
    if driver_ahead:
        title += f" vs {driver_ahead}"

    fig.update_layout(
        title=title,
        xaxis_title="Lap Number",
        yaxis_title="Overtake Probability",
        height=400,
        yaxis=dict(range=[0, 1]),
    )

    return fig


def create_confusion_matrix_plot(metrics: dict, model_name: str = "") -> go.Figure:
    """Create confusion matrix visualization.

    Args:
        metrics: Metrics dictionary with tn, fp, fn, tp
        model_name: Model name for title

    Returns:
        Plotly figure
    """
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted Negative", "Predicted Positive"],
            y=["Actual Negative", "Actual Positive"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )

    fig.update_layout(
        title=f"Confusion Matrix: {model_name}",
        height=400,
    )

    return fig


# Import for race timeline plot
from f1overtake.split import prepare_xy
