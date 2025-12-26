# Quick Start Guide - F1 Overtake Probability Model

**Author:** João Pedro Cunha

## Installation

```bash
# Clone repository
cd /Users/jpcunha/Documents/Portfolio/f1-overtake-probability

# Install dependencies
pip install optuna  # New dependency for hyperparameter tuning

# Or use poetry
poetry install
```

## 5-Minute Quick Start

### 1. Train a Basic Model (Quick Mode)
```bash
python -m f1overtake.train --quick
```
**What it does:** Trains on 2 races (~1 min)

### 2. Train Production Model with Tuning
```bash
python -m f1overtake.train --tune
```
**What it does:**
- Trains on all 24 races
- Runs Optuna hyperparameter search (50 trials)
- Generates calibrated models
- Creates HTML reports
- Time: ~45 min

### 3. Score a Specific Race
```bash
f1overtake score-race --year 2024 --event "Monaco" --driver "VER" --output outputs/monaco
```
**What it does:**
- Loads trained models
- Generates per-lap probabilities
- Calculates xO metric
- Exports CSV files

### 4. Launch Interactive App
```bash
streamlit run app/streamlit_app.py
```
**What it does:**
- Opens browser at localhost:8501
- Interactive model exploration
- xO leaderboard
- Prediction visualizations

## Common Operations

### Training

#### Basic Training
```bash
python -m f1overtake.train
```

#### With Hyperparameter Tuning
```bash
python -m f1overtake.train --tune
```

#### Quick Mode (Testing)
```bash
python -m f1overtake.train --quick
```

### Evaluation

#### Evaluate All Models
```bash
f1overtake evaluate
```

#### Generate HTML Reports
```bash
f1overtake evaluate --report
```

### Batch Scoring

#### Score Entire Race
```bash
f1overtake score-race --event "Bahrain"
```

#### Score Specific Driver
```bash
f1overtake score-race --event "Monaco" --driver "VER"
```

#### Save Results
```bash
f1overtake score-race --event "Italy" --output outputs/italy_results
```

#### Use Specific Model
```bash
f1overtake score-race --event "Spain" --model xgboost_calibrated
```

## Configuration

### Enable Temporal Features
```python
from f1overtake.config import Config

config = Config()
config.features.enable_temporal_features = True
config.features.lagged_gap_laps = [1, 2]
config.features.closing_rate_window = 3
```

### Configure Optuna Tuning
```python
config.model.use_optuna = True
config.model.optuna_n_trials = 50
config.model.optuna_cv_folds = 5
config.model.optuna_timeout = 3600  # 1 hour
```

### Adjust Performance Targets
```python
config.model.target_roc_auc = 0.75
config.model.target_pr_auc_multiplier = 2.0  # 2x baseline
```

## Python API

### Calculate xO Metric
```python
from f1overtake.xo_metric import calculate_xo, create_xo_leaderboard
from f1overtake.train import load_models
from f1overtake.build_dataset import build_dataset

# Load model
models = load_models("models")
model = models["xgboost_calibrated"]

# Load data
dataset = build_dataset()

# Calculate xO
xo_df = calculate_xo(dataset, model)

# Create leaderboard
leaderboard = create_xo_leaderboard(xo_df, race_name="Bahrain", top_n=20)
print(leaderboard)
```

### Generate HTML Report
```python
from f1overtake.model_report import generate_html_report
from f1overtake.split import split_by_race

# Split data
train_df, test_df = split_by_race(dataset, test_size=0.3)

# Generate report
generate_html_report(
    model=model,
    model_name="xgboost_calibrated",
    train_df=train_df,
    test_df=test_df,
    config=config,
    output_path="outputs/model_report.html"
)
```

### Run Hyperparameter Tuning
```python
from f1overtake.tune import tune_xgboost

# Tune XGBoost
best_params = tune_xgboost(
    train_df,
    config,
    save_path="models/best_params.json"
)

print(f"Best parameters: {best_params}")
```

### Label Sensitivity Analysis
```python
from f1overtake.labels import label_sensitivity_analysis

# Analyze label sensitivity
sensitivity = label_sensitivity_analysis(
    lap_data,
    lookahead_values=[1, 2, 3],
    config=label_config
)

print(sensitivity)
```

## File Locations

### Models
- **Trained models:** `models/*.pkl`
- **Best parameters:** `models/best_params.json`
- **Calibrated models:** `models/*_calibrated.pkl`

### Outputs
- **HTML reports:** `outputs/*_report.html`
- **Batch scores:** `outputs/*_predictions.csv`
- **xO tables:** `outputs/*_xo.csv`

### Data
- **Cached data:** `cache/`
- **Built datasets:** `datasets/`

## Streamlit App Tabs

1. **Dataset & Training:** Build datasets and train models
2. **Model Evaluation:** View performance metrics and visualizations
3. **Predictions:** What-if scenario analysis and race timelines
4. **xO Leaderboard:** Driver performance analysis with xO metric
5. **About:** Model documentation and limitations

## Performance Expectations

### Training
- **Quick mode (2 races):** 30-60 seconds
- **Full mode (24 races):** 2-3 minutes
- **With tuning (50 trials):** 30-45 minutes

### Prediction
- **Single race:** < 1 second
- **All races:** 2-3 seconds

### Expected Metrics
- **ROC AUC:** 0.78-0.82
- **PR AUC:** 2.5-3.0x baseline
- **Brier Score:** 0.06-0.08
- **ECE:** 0.04-0.06

## Troubleshooting

### Import Error: optuna
```bash
pip install optuna
```

### Temporal Features All NaN
- Check sequential laps in data
- Verify `lagged_gap_laps` values

### Poor Model Performance
- Try `--tune` flag
- Increase training data
- Check data quality

### xO Values Seem Off
- Check model calibration (use calibrated model)
- Verify `min_opportunities` threshold
- Review Brier score and ECE

## Tips for Best Results

1. **Always use calibrated models** for probability estimates
2. **Enable temporal features** for better performance
3. **Run hyperparameter tuning** for production models
4. **Generate HTML reports** to document model quality
5. **Monitor xO metric** for business insights
6. **Use PR AUC** as primary metric (not ROC AUC)
7. **Check calibration quality** (ECE, Brier score)

## Next Steps

1. **Read:** `PRODUCTION_GUIDE.md` for detailed documentation
2. **Explore:** Streamlit app for interactive analysis
3. **Review:** HTML model reports in `outputs/`
4. **Experiment:** Try different configurations
5. **Deploy:** Follow production checklist in guide

## Support

For issues or questions, refer to:
- `PRODUCTION_GUIDE.md`: Detailed documentation
- `CHANGELOG_PRODUCTION.md`: List of changes
- `README.md`: Project overview

---

**Author:** João Pedro Cunha
**Last Updated:** December 26, 2025
