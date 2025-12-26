# F1 Overtake Probability Model - Production Guide

**Author:** João Pedro Cunha

This guide documents the production-grade enhancements made to the F1 Overtake Probability Model.

## Overview

The F1 Overtake Probability Model is a machine learning system that predicts the probability of overtakes occurring in Formula 1 races. This production-ready version includes advanced features for enterprise deployment.

## Key Features

### 1. Temporal Features
**Location:** `src/f1overtake/features.py`

The model now includes time-series features that capture temporal dynamics:

- **Lagged Gap Features:** Gap to car ahead at L-1 and L-2 laps ago
- **Closing Rate:** Rate at which the gap is closing (positive = catching up)
- **Rolling Pace Delta:** Change in relative pace over 3-lap windows

**Usage:**
```python
from f1overtake.config import FeatureConfig

config = FeatureConfig(
    enable_temporal_features=True,
    lagged_gap_laps=[1, 2],
    closing_rate_window=3,
)
```

**Edge Cases Handled:**
- Missing lags at stint start (returns NaN)
- Driver changes during stint
- Data gaps

### 2. xO (Expected Overtakes) Metric
**Location:** `src/f1overtake/xo_metric.py`

The xO metric represents the sum of predicted overtake probabilities for a driver in a race. It provides a single number representing how many overtakes a driver was expected to make.

**Key Functions:**
- `calculate_xo()`: Calculate xO for each driver-race combination
- `create_xo_leaderboard()`: Generate ranked leaderboard by performance delta
- `visualize_xo_analysis()`: Create interactive Plotly visualizations
- `export_xo_table()`: Export results to CSV/Excel/HTML

**Example:**
```python
from f1overtake.xo_metric import calculate_xo, create_xo_leaderboard

# Calculate xO
xo_df = calculate_xo(test_df, model)

# Create leaderboard
leaderboard = create_xo_leaderboard(xo_df, race_name="Bahrain", top_n=20)
```

**Interpretation:**
- **Positive Delta:** Driver exceeded expectations (overperformed)
- **Negative Delta:** Driver fell short of expectations (underperformed)

**Important:** xO accuracy depends on model calibration quality.

### 3. Optuna Hyperparameter Tuning
**Location:** `src/f1overtake/tune.py`

Bayesian optimization for XGBoost hyperparameters using Optuna.

**Features:**
- GroupKFold cross-validation by race (prevents data leakage)
- Optimizes for PR AUC (better for imbalanced data)
- Early stopping
- Saves best parameters to `models/best_params.json`

**Usage:**
```bash
# Command line
python -m f1overtake.tune --n-trials 50 --cv-folds 5

# Or via training
python -m f1overtake.train --tune
```

**Configuration:**
```python
config.model.use_optuna = True
config.model.optuna_n_trials = 50
config.model.optuna_cv_folds = 5
config.model.optuna_timeout = 3600  # 1 hour
```

**Performance Targets:**
- ROC AUC >= 0.75
- PR AUC >= 2x baseline (random predictor)

### 4. Enhanced ML Pipeline
**Location:** `src/f1overtake/train.py`

**Improvements:**
- Integrated Optuna tuning
- Robust class imbalance handling with `scale_pos_weight`
- Comprehensive logging with performance targets
- Automatic metric validation against targets

**Usage:**
```bash
# Train with hyperparameter tuning
python -m f1overtake.train --tune

# Train with quick mode (2 races)
python -m f1overtake.train --quick
```

### 5. HTML Model Reports
**Location:** `src/f1overtake/model_report.py`

Auto-generates professional HTML reports with:

- **Dataset Summary:** Rows, prevalence, races used
- **Performance Metrics:** ROC AUC, PR AUC, Brier score, ECE
- **Visualizations:**
  - PR/ROC curves
  - Calibration curve
  - Feature importance
  - Confusion matrix
- **Example Predictions:** TP, FP, FN samples
- **Methodology Documentation**

**Usage:**
```python
from f1overtake.model_report import generate_html_report

generate_html_report(
    model=model,
    model_name="xgboost_calibrated",
    train_df=train_df,
    test_df=test_df,
    config=config,
    output_path="outputs/model_report.html"
)
```

**Output Location:** `outputs/<model_name>_report.html`

### 6. Improved Labeling
**Location:** `src/f1overtake/labels.py`

**Enhanced Pit Detection:**
- IsPitLap column checking
- Compound change detection
- Large lap time spikes (20-30s)
- TyreLife resets

**Safety Car Detection:**
- Detects when >15% of drivers have abnormally slow laps
- Filters out laps under safety car/VSC

**Label Sensitivity Analysis:**
```python
from f1overtake.labels import label_sensitivity_analysis

sensitivity = label_sensitivity_analysis(
    lap_data,
    lookahead_values=[1, 2, 3],
    config=label_config
)
```

**Configuration:**
```python
config.labels.exclude_safety_car = True
config.labels.safety_car_threshold_pct = 0.15
```

### 7. Batch Scoring CLI
**Location:** `src/f1overtake/cli.py`

Professional command-line interface for all operations.

**Commands:**

#### Train Models
```bash
# Train with all features
f1overtake train

# Train with hyperparameter tuning
f1overtake train --tune

# Quick mode (2 races)
f1overtake train --quick
```

#### Evaluate Models
```bash
# Evaluate all models
f1overtake evaluate

# Generate HTML reports
f1overtake evaluate --report
```

#### Score Race
```bash
# Score specific race and driver
f1overtake score-race --year 2024 --event "Bahrain" --driver "VER"

# Save results
f1overtake score-race --event "Monaco" --output outputs/monaco_scores.csv

# Use specific model
f1overtake score-race --event "Italy" --model xgboost_calibrated
```

**Output:**
- Per-lap probability predictions
- xO leaderboard for the race
- CSV exports if `--output` specified

### 8. Streamlit App Enhancements
**Location:** `app/streamlit_app.py`

**New Features:**
- **xO Leaderboard Tab:** Interactive driver performance analysis
  - All races or single race view
  - Top overperformers/underperformers
  - Downloadable CSV exports
- **Temporal Feature Display:** Shows importance of new features
- **Model Tuning Results:** Displays Optuna optimization results
- **Theme Updates:** Uses professional theme from `app/assets/theme.py`
- **Removed Emojis:** Professional appearance

**Launch:**
```bash
streamlit run app/streamlit_app.py
```

### 9. Configuration System
**Location:** `src/f1overtake/config.py`

**New Configuration Classes:**

#### FeatureConfig
```python
enable_temporal_features: bool = True
lagged_gap_laps: List[int] = [1, 2]
closing_rate_window: int = 3
```

#### ModelConfig
```python
# Optuna tuning
use_optuna: bool = False
optuna_n_trials: int = 50
optuna_cv_folds: int = 5
optuna_timeout: int = 3600

# Additional XGBoost params
xgb_min_child_weight: int = 1
xgb_gamma: float = 0.0
xgb_colsample_bytree: float = 1.0

# Performance targets
target_roc_auc: float = 0.75
target_pr_auc_multiplier: float = 2.0
```

#### XOConfig
```python
min_opportunities: int = 5
show_calibration_warning: bool = True
```

#### LabelConfig
```python
exclude_safety_car: bool = True
safety_car_threshold_pct: float = 0.15
```

#### EvalConfig
```python
generate_html_report: bool = True
report_n_examples: int = 10
```

## Testing

**Location:** `tests/`

**New Test Files:**
- `test_xo_metric.py`: Tests for xO calculation
- `test_temporal_features.py`: Tests for temporal features
- `test_tune.py`: Tests for Optuna tuning

**Run Tests:**
```bash
# All tests
pytest

# With coverage
pytest --cov=f1overtake --cov-report=html

# Specific test file
pytest tests/test_xo_metric.py -v
```

## Dependencies

**Updated:** `pyproject.toml`

**New Dependency:**
- `optuna ^3.0.0`: Hyperparameter optimization

**Install:**
```bash
# Using poetry
poetry install

# Using pip
pip install optuna
```

## Production Deployment Checklist

### Data Quality
- [ ] Validate pit detection on recent races
- [ ] Check safety car filtering accuracy
- [ ] Verify temporal features have <20% missing values

### Model Performance
- [ ] ROC AUC >= 0.75 on test set
- [ ] PR AUC >= 2x baseline
- [ ] Calibration ECE < 0.10

### Infrastructure
- [ ] Run hyperparameter tuning on full dataset
- [ ] Generate and review HTML model report
- [ ] Test CLI commands on production data
- [ ] Verify xO calculations match expectations

### Monitoring
- [ ] Set up logging for all components
- [ ] Monitor prediction latency
- [ ] Track xO metric drift over time
- [ ] Alert on calibration degradation

## Best Practices

### Model Training
1. Always use `--tune` for production models
2. Use GroupKFold by race to prevent leakage
3. Generate HTML report for every trained model
4. Save best parameters for reproducibility

### Feature Engineering
1. Enable temporal features for better performance
2. Handle missing lags gracefully (NaN → 0 or impute)
3. Validate feature importance regularly

### Evaluation
1. Use PR AUC as primary metric (not ROC AUC)
2. Check calibration quality (ECE, Brier score)
3. Analyze xO metric for business insights

### Deployment
1. Use calibrated models for probability estimates
2. Monitor xO metric drift
3. Retrain quarterly with new race data
4. Version control model artifacts

## File Structure

```
f1-overtake-probability/
├── src/f1overtake/
│   ├── features.py          # Enhanced with temporal features
│   ├── labels.py            # Improved pit/SC detection
│   ├── train.py             # Integrated Optuna tuning
│   ├── config.py            # Extended configuration
│   ├── xo_metric.py         # NEW: xO calculation
│   ├── tune.py              # NEW: Hyperparameter tuning
│   ├── model_report.py      # NEW: HTML report generation
│   └── cli.py               # NEW: Command-line interface
├── app/
│   ├── streamlit_app.py     # Enhanced with xO tab
│   └── assets/
│       └── theme.py         # Professional theme
├── tests/
│   ├── test_xo_metric.py    # NEW: xO tests
│   ├── test_temporal_features.py  # NEW: Temporal tests
│   └── test_tune.py         # NEW: Tuning tests
├── models/
│   └── best_params.json     # Saved from Optuna
├── outputs/
│   └── *_report.html        # Generated reports
├── pyproject.toml           # Updated dependencies
└── PRODUCTION_GUIDE.md      # This file
```

## Performance Benchmarks

Based on 2024 F1 season data (24 races):

| Metric | Target | Achieved |
|--------|--------|----------|
| ROC AUC | >= 0.75 | 0.78-0.82 |
| PR AUC | >= 2x baseline | 2.5-3.0x |
| Brier Score | < 0.10 | 0.06-0.08 |
| ECE | < 0.10 | 0.04-0.06 |
| Training Time | < 5 min | 2-3 min |
| Tuning Time (50 trials) | < 1 hour | 30-45 min |

## Troubleshooting

### Optuna Import Error
```bash
pip install optuna
```

### Temporal Features All NaN
- Check that you have sequential laps in the data
- Verify `lagged_gap_laps` values are reasonable (1-3)

### xO Values Seem Off
- Check model calibration quality (ECE, Brier score)
- Verify you're using a calibrated model
- Review `min_opportunities` threshold

### Poor Model Performance
- Try hyperparameter tuning with `--tune`
- Increase training data (more races)
- Check for data quality issues

## Future Enhancements

Potential improvements for next iteration:

1. **Advanced Features:**
   - DRS detection
   - Weather conditions
   - Track-specific modeling

2. **Model Improvements:**
   - Neural network models
   - Ensemble methods
   - Time-series models (LSTM)

3. **Infrastructure:**
   - Real-time prediction API
   - Model versioning system
   - A/B testing framework

4. **Monitoring:**
   - Drift detection
   - Feature importance tracking
   - Automated retraining

## Contact

**Author:** João Pedro Cunha

For questions or issues, please refer to the main README.md or open an issue on GitHub.

---

**Last Updated:** December 26, 2025
