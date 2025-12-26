"""Professional HTML model report generation.

Author: João Pedro Cunha

Generates a comprehensive HTML report with:
- Dataset summary
- Feature importance
- Model performance metrics
- Calibration curves
- Example predictions
- Methodology documentation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from f1overtake.config import Config
from f1overtake.evaluate import get_feature_importance
from f1overtake.split import prepare_xy

logger = logging.getLogger(__name__)


def calculate_ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE).

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins

    Returns:
        ECE score
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_pred_proba[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def create_pr_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
) -> str:
    """Create PR and ROC curves as HTML.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Model name for title

    Returns:
        HTML string with plots
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Precision-Recall Curve", "ROC Curve"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR AUC = {pr_auc:.3f}",
            line=dict(color="blue", width=2),
        ),
        row=1,
        col=1,
    )

    # Baseline
    baseline = y_true.mean()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[baseline, baseline],
            mode="lines",
            name=f"Baseline = {baseline:.3f}",
            line=dict(color="gray", dash="dash"),
        ),
        row=1,
        col=1,
    )

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC AUC = {roc_auc:.3f}",
            line=dict(color="green", width=2),
        ),
        row=1,
        col=2,
    )

    # Diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", dash="dash"),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Recall", row=1, col=1)
    fig.update_yaxes(title_text="Precision", row=1, col=1)
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)

    fig.update_layout(height=400, showlegend=True)

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_calibration_plot_html(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> str:
    """Create calibration plot as HTML.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins

    Returns:
        HTML string with plot
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_sums[i] = y_true[mask].sum()
            bin_counts[i] = mask.sum()

    bin_means = np.where(bin_counts > 0, bin_sums / bin_counts, 0)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()

    # Calibration curve
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=bin_means,
            mode="markers+lines",
            name="Calibration",
            marker=dict(size=10, color="blue"),
            line=dict(color="blue", width=2),
        )
    )

    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="gray", dash="dash"),
        )
    )

    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Frequency",
        height=400,
        showlegend=True,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def create_feature_importance_html(importance_df: pd.DataFrame) -> str:
    """Create feature importance plot as HTML.

    Args:
        importance_df: DataFrame with feature importances

    Returns:
        HTML string with plot
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
            marker=dict(color="steelblue"),
        )
    )

    fig.update_layout(
        title="Top Feature Importance",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, len(importance_df) * 30),
        showlegend=False,
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def get_example_predictions(
    df: pd.DataFrame,
    model,
    n_examples: int = 10,
) -> Dict[str, pd.DataFrame]:
    """Get example predictions for each category (TP, FP, FN, TN).

    Args:
        df: Test DataFrame
        model: Trained model
        n_examples: Number of examples per category

    Returns:
        Dictionary with example DataFrames for each category
    """
    X, y_true = prepare_xy(df)

    if hasattr(model, "feature_names_"):
        X = X[model.feature_names_]

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Add predictions to df
    results = df.copy()
    results["Probability"] = y_pred_proba
    results["Predicted"] = y_pred

    # Categorize
    tp = results[(results["Overtake"] == 1) & (results["Predicted"] == 1)]
    fp = results[(results["Overtake"] == 0) & (results["Predicted"] == 1)]
    fn = results[(results["Overtake"] == 1) & (results["Predicted"] == 0)]
    tn = results[(results["Overtake"] == 0) & (results["Predicted"] == 0)]

    # Get top examples
    examples = {
        "True Positives": tp.nlargest(n_examples, "Probability")[
            ["RaceName", "Driver", "DriverAhead", "Gap", "Probability", "Overtake"]
        ],
        "False Positives": fp.nlargest(n_examples, "Probability")[
            ["RaceName", "Driver", "DriverAhead", "Gap", "Probability", "Overtake"]
        ],
        "False Negatives": fn.nsmallest(n_examples, "Probability")[
            ["RaceName", "Driver", "DriverAhead", "Gap", "Probability", "Overtake"]
        ],
    }

    return examples


def generate_html_report(
    model,
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
    output_path: str,
    run_id: Optional[str] = None,
) -> None:
    """Generate comprehensive HTML model report.

    Args:
        model: Trained model
        model_name: Model name
        train_df: Training DataFrame
        test_df: Test DataFrame
        config: Configuration
        output_path: Output file path
        run_id: Optional run identifier
    """
    logger.info(f"Generating HTML report for {model_name}...")

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare test data
    X_test, y_test = prepare_xy(test_df)
    if hasattr(model, "feature_names_"):
        X_test = X_test[model.feature_names_]

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= config.eval.decision_threshold).astype(int)

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    brier = brier_score_loss(y_test, y_pred_proba)
    ece = calculate_ece(y_test, y_pred_proba, config.eval.n_bins)
    cm = confusion_matrix(y_test, y_pred)

    # Get feature importance
    importance_df = get_feature_importance(model, top_n=15)

    # Get example predictions
    examples = get_example_predictions(test_df, model, config.eval.report_n_examples)

    # Create visualizations
    pr_roc_html = create_pr_roc_curves(y_test, y_pred_proba, model_name)
    calibration_html = create_calibration_plot_html(y_test, y_pred_proba, config.eval.n_bins)
    importance_html = (
        create_feature_importance_html(importance_df) if not importance_df.empty else ""
    )

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>F1 Overtake Probability Model Report - {model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .confusion-matrix {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            max-width: 400px;
            margin: 20px auto;
        }}
        .cm-cell {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            border-radius: 8px;
            border: 2px solid #ddd;
        }}
        .cm-cell.tp {{ border-color: #28a745; }}
        .cm-cell.fp {{ border-color: #ffc107; }}
        .cm-cell.fn {{ border-color: #dc3545; }}
        .cm-cell.tn {{ border-color: #6c757d; }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>F1 Overtake Probability Model Report</h1>
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Run ID:</strong> {run_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Author:</strong> João Pedro Cunha</p>
    </div>

    <div class="section">
        <h2>Dataset Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{len(train_df) + len(test_df):,}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(train_df):,}</div>
                <div class="metric-label">Training Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(test_df):,}</div>
                <div class="metric-label">Test Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{test_df["Overtake"].mean()*100:.1f}%</div>
                <div class="metric-label">Overtake Prevalence</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{train_df["RaceName"].nunique()}</div>
                <div class="metric-label">Training Races</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{test_df["RaceName"].nunique()}</div>
                <div class="metric-label">Test Races</div>
            </div>
        </div>

        <p><strong>Test Races:</strong> {', '.join(sorted(test_df['RaceName'].unique()))}</p>
    </div>

    <div class="section">
        <h2>Model Performance</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{roc_auc:.3f}</div>
                <div class="metric-label">ROC AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{pr_auc:.3f}</div>
                <div class="metric-label">PR AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{brier:.3f}</div>
                <div class="metric-label">Brier Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{ece:.3f}</div>
                <div class="metric-label">ECE</div>
            </div>
        </div>

        {pr_roc_html}
    </div>

    <div class="section">
        <h2>Calibration</h2>
        <p>A well-calibrated model's predicted probabilities should match actual frequencies.</p>
        {calibration_html}

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{brier:.4f}</div>
                <div class="metric-label">Brier Score (lower is better)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{ece:.4f}</div>
                <div class="metric-label">Expected Calibration Error</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Confusion Matrix</h2>
        <div class="confusion-matrix">
            <div class="cm-cell tp">
                <div style="font-size: 24px; font-weight: bold;">{cm[1][1]}</div>
                <div style="color: #28a745;">True Positives</div>
            </div>
            <div class="cm-cell fp">
                <div style="font-size: 24px; font-weight: bold;">{cm[0][1]}</div>
                <div style="color: #ffc107;">False Positives</div>
            </div>
            <div class="cm-cell fn">
                <div style="font-size: 24px; font-weight: bold;">{cm[1][0]}</div>
                <div style="color: #dc3545;">False Negatives</div>
            </div>
            <div class="cm-cell tn">
                <div style="font-size: 24px; font-weight: bold;">{cm[0][0]}</div>
                <div style="color: #6c757d;">True Negatives</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Feature Importance</h2>
        {importance_html if importance_html else '<p>Feature importance not available for this model.</p>'}
    </div>

    <div class="section">
        <h2>Example Predictions</h2>

        <h3>True Positives (Correctly Predicted Overtakes)</h3>
        {examples['True Positives'].to_html(index=False, classes='table') if len(examples['True Positives']) > 0 else '<p>No examples available.</p>'}

        <h3>False Positives (Incorrectly Predicted Overtakes)</h3>
        {examples['False Positives'].to_html(index=False, classes='table') if len(examples['False Positives']) > 0 else '<p>No examples available.</p>'}

        <h3>False Negatives (Missed Overtakes)</h3>
        {examples['False Negatives'].to_html(index=False, classes='table') if len(examples['False Negatives']) > 0 else '<p>No examples available.</p>'}
    </div>

    <div class="section">
        <h2>Methodology</h2>

        <h3>Train/Test Split</h3>
        <p>To prevent data leakage, the dataset was split by race weekend (not randomly). This ensures the model is tested on completely unseen races, providing a realistic assessment of generalization performance.</p>

        <h3>Class Imbalance</h3>
        <p>Overtakes are rare events (~{test_df["Overtake"].mean()*100:.1f}% prevalence). We handle this imbalance using:</p>
        <ul>
            <li>Scale position weight in XGBoost</li>
            <li>PR AUC as the primary metric (more informative than ROC AUC for imbalanced data)</li>
            <li>Calibration techniques to ensure reliable probability estimates</li>
        </ul>

        <h3>Features</h3>
        <p>The model uses the following feature categories:</p>
        <ul>
            <li><strong>Gap features:</strong> Time gap to car ahead</li>
            <li><strong>Pace features:</strong> Rolling average lap times, relative pace</li>
            <li><strong>Tire features:</strong> Tire age, compound differences</li>
            <li><strong>Temporal features:</strong> Lagged gaps, closing rates, rolling pace deltas</li>
            <li><strong>Track features:</strong> Track-specific encoding</li>
            <li><strong>Race phase features:</strong> Lap number, race progress</li>
        </ul>

        <div class="warning">
            <strong>Limitations:</strong>
            <ul>
                <li>Position changes may occur without on-track overtakes (pit strategy)</li>
                <li>Label noise exists due to incomplete pit detection</li>
                <li>Safety car periods may not be fully filtered</li>
                <li>DRS and track-specific factors not explicitly modeled</li>
                <li>Weather conditions and traffic not considered</li>
            </ul>
        </div>
    </div>

    <div class="footer">
        <p>F1 Overtake Probability Model Report</p>
        <p>Author: João Pedro Cunha</p>
        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
</body>
</html>
"""

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"HTML report saved to {output_path}")
