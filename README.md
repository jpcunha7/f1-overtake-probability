# F1 Overtake Probability Model: "Can We Pass?"

**Machine learning system for predicting overtaking opportunities in Formula 1 races**

**Author:** João Pedro Cunha
**License:** MIT

---

## Overview

This project uses supervised machine learning to predict the probability of an overtake occurring in F1 races based on:
- **Gap to car ahead** (time difference)
- **Relative pace** (recent lap time comparison)
- **Tire age and compound** differences
- **Race phase** (early, middle, late)
- **Track context** (circuit characteristics)

**Key Features:**
- **Supervised ML Pipeline**: Logistic Regression + XGBoost models
- **Calibrated Probabilities**: Isotonic/Platt calibration for reliable predictions
- **Explainability**: Feature importance analysis
- **Interactive App**: Streamlit dashboard for scenario exploration
- **Robust Evaluation**: ROC AUC, PR AUC, Brier score, calibration curves

**100% Free**: Uses only open-source tools and FastF1 data.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- Poetry (recommended) or pip

### Using Poetry

```bash
git clone https://github.com/jpcunha7/f1-overtake-probability.git
cd f1-overtake-probability
poetry install
poetry shell
```

---

## Quick Start

### 1. Build Dataset (Quick Mode)

```bash
python -m f1overtake.build_dataset --quick
```

Builds a dataset from 2-3 recent races (~5 minutes).

### 2. Train Models

```bash
python -m f1overtake.train --quick
```

Trains Logistic Regression and XGBoost models with calibration.

### 3. Launch Streamlit App

```bash
poetry run streamlit run app/streamlit_app.py
```

Open http://localhost:8501 to explore the interactive dashboard.

---

## Methodology

### 1. Overtake Labeling Strategy

**Definition:**
An overtake opportunity exists when:
1. Driver `i` is directly behind driver `j` on lap `L`
2. We check if driver `i` is ahead of driver `j` within the next `K` laps (default K=1)

**Label:**
- `y = 1` if overtake occurs
- `y = 0` otherwise

**Pit Confounding Mitigation:**

Pit stops create artificial position changes. We filter these using:

- **Exclude laps where either driver pits** in current lap or lookahead window
- **Exclude laps with compound changes** (indicates pit stop)
- **Exclude outlier lap times** (formation laps, safety cars)
- **Minimum stint length requirement** (ensures clean data)

### 2. Feature Engineering

**Gap Features:**
- Time gap to car ahead
- Gap categorical bins (very_close, close, medium, far)

**Pace Features:**
- Rolling average lap times (3-lap window)
- Relative pace (difference in recent lap times)
- Pace ratio

**Tire Features:**
- Tire age difference (fresher tires = advantage)
- Compound difference (SOFT > MEDIUM > HARD)
- Same compound flag

**Race Phase Features:**
- Lap number / race progress
- Race phase (early/middle/late)

**Track Features:**
- Circuit identifier (one-hot encoded)

### 3. Model Training

**Baseline: Logistic Regression**
- Simple, interpretable
- Class weights for imbalance handling

**Advanced: XGBoost**
- Gradient boosting classifier
- Handles non-linear relationships
- Feature interactions
- Scale_pos_weight for imbalance

**Train/Test Split:**
- Split by **race weekend** (not random) to avoid leakage
- Test on completely unseen races
- Typical split: 70% train, 30% test

### 4. Probability Calibration

**Why Calibration Matters:**

Raw model probabilities may not reflect true frequencies. For example:
- Model predicts 70% → actual overtake rate might be 50%

**Calibration Methods:**
- **Isotonic Regression**: Non-parametric, flexible (default)
- **Platt Scaling**: Parametric, sigmoid-based

**After calibration:** Predicted 70% → ~70% of cases result in overtakes

### 5. Evaluation Metrics

**Classification Metrics:**
- **ROC AUC**: Overall discrimination ability
- **PR AUC**: Precision-recall trade-off (better for imbalanced data)
- **Brier Score**: Measures probability calibration quality (lower is better)

**Calibration:**
- **Calibration Curve**: Predicted vs actual frequencies
- **Histogram**: Distribution of predicted probabilities

**Feature Importance:**
- **Permutation Importance**: Feature impact on model performance
- **SHAP** (optional): Individual prediction explanations

---

## Modeling Assumptions & Limitations

### What We Model

- Tire degradation differences
- Gap and relative pace
- Pit stop filtering (heuristic-based)
- Probability calibration
- Race phase effects

### What We DON'T Model

- **DRS (Drag Reduction System)**: Major overtaking aid not explicitly modeled
- **Track-specific characteristics**: Overtaking difficulty varies by circuit
- **Traffic and backmarkers**: Multi-car interactions ignored
- **Weather conditions**: Rain/wet races not handled
- **Safety cars and VSC**: Some SC periods may leak through filtering
- **Driver skill differences**: Assumes all drivers equal
- **Incident-based position changes**: Crashes, mechanical failures
- **Perfect pit detection**: Label noise exists despite filtering

### Interpretation Guidelines

**Do:**
- Use for comparative analysis ("How do gap and tire age affect overtake probability?")
- Explore scenario sensitivity (what-if analysis)
- Understand model limitations and uncertainty

**Don't:**
- Treat predictions as absolute truth
- Ignore label noise and confounding factors
- Use for betting or real-time race strategy (insufficient real-time data)

**Transparency:** This is a **proxy model** with **label noise**. Position changes ≠ on-track overtakes.

---

## Project Structure

```
f1-overtake-probability/
├── README.md
├── LICENSE (MIT)
├── pyproject.toml
├── src/f1overtake/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── data_loader.py          # FastF1 data loading
│   ├── labels.py               # Overtake labeling logic
│   ├── features.py             # Feature engineering
│   ├── build_dataset.py        # Dataset construction
│   ├── split.py                # Train/test splitting
│   ├── train.py                # Model training
│   ├── calibrate.py            # Probability calibration
│   ├── evaluate.py             # Model evaluation
│   └── viz.py                  # Visualizations
├── app/
│   └── streamlit_app.py        # Interactive Streamlit dashboard
├── tests/
│   ├── test_labels.py
│   └── test_features.py
└── .github/workflows/
    └── ci.yml                  # GitHub Actions CI
```

---

## Examples

### Example 1: Build Full Dataset

```bash
python -m f1overtake.build_dataset
```

Builds dataset from all configured races in `config.py`.

### Example 2: Train with Custom Config

```python
from f1overtake.config import Config, DataConfig
from f1overtake.build_dataset import build_dataset
from f1overtake.train import train_all_models

# Custom configuration
config = Config(
    data=DataConfig(
        year=2024,
        race_names=["Bahrain", "Saudi Arabia", "Australia", "Japan"]
    )
)

# Build and train
dataset = build_dataset(config)
models = train_all_models(dataset, config)
```

### Example 3: Interactive Streamlit App

```bash
poetry run streamlit run app/streamlit_app.py
```

**Features:**
- Dataset building and summary
- Model training and evaluation
- Calibration curves and confusion matrices
- Feature importance visualization
- What-if scenario analysis with sliders
- Race timeline probability plots

---

## Visualizations

The tool generates:

1. **Calibration Curve**: Predicted vs actual overtake frequencies
2. **Feature Importance**: Bar chart of top predictive features
3. **Probability Heatmap**: Overtake probability vs gap and tire age
4. **Race Timeline**: Lap-by-lap probability for selected driver
5. **Confusion Matrix**: Classification performance
6. **ROC/PR Curves**: Model discrimination ability (in evaluation scripts)

---

## Development

### Running Tests

```bash
poetry run pytest
poetry run pytest --cov=f1overtake --cov-report=term-missing
```

### Code Quality

```bash
poetry run black src/ tests/
poetry run ruff check src/ tests/
```

### CI/CD

GitHub Actions runs tests on Python 3.10, 3.11, 3.12 for every push/PR.

---

## Data Source

**FastF1** (https://docs.fastf1.dev/)
- Free, open-source Python package
- F1 timing, telemetry, and race data
- Supports 2018-2025 seasons
- Data cached locally to avoid re-downloads

---

## Troubleshooting

### "No overtake opportunities found"
- Try different races (some have more overtaking)
- Check `max_gap` parameter (increase if too restrictive)
- Verify race data loaded successfully

### "Model performance is poor"
- This is expected! Overtakes are inherently unpredictable
- Typical ROC AUC: 0.60-0.75 (better than random but not perfect)
- High class imbalance (most opportunities don't result in overtakes)

### "Calibration curve shows poor calibration"
- Try different calibration method (isotonic vs sigmoid)
- Increase calibration dataset size
- Check for distribution shift between train/test races

---

## Roadmap

Future enhancements:
- [ ] DRS zone modeling
- [ ] Track-specific overtaking difficulty
- [ ] Multi-lap lookahead (overtake within 3-5 laps)
- [ ] Weather/rain handling
- [ ] Real-time prediction API
- [ ] SHAP explainability integration

---

## Acknowledgments

- **FastF1**: Free F1 data access
- **scikit-learn**: Machine learning framework
- **XGBoost**: Gradient boosting implementation
- **Plotly & Streamlit**: Interactive visualizations

---

## Citation

If you use this tool in research or presentations:

```
Cunha, J.P. (2025). F1 Overtake Probability Model: Supervised ML for Overtaking Prediction.
https://github.com/jpcunha7/f1-overtake-probability
```
