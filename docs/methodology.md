# F1 Overtake Probability Model: Methodology

**Author:** João Pedro Cunha

---

## Overview

This document describes the machine learning methodology, labeling strategy, feature engineering, and evaluation approach for predicting Formula 1 overtaking probabilities.

## 1. Problem Formulation

### Supervised Learning Task
- **Type:** Binary classification
- **Positive class (y=1):** Overtake occurs
- **Negative class (y=0):** No overtake
- **Prediction:** Calibrated probability P(overtake | features)

### Definition of Overtake Opportunity
An overtake opportunity exists when:
1. Driver i is directly behind driver j on lap L (consecutive positions)
2. We observe whether driver i overtakes driver j within K laps (default K=1)

## 2. Labeling Strategy

### Basic Label Logic
```
For each lap L and driver pair (i, j) where position_i(L) = position_j(L) + 1:
  if position_i(L+K) < position_j(L+K):
    label = 1  # Overtake occurred
  else:
    label = 0  # No overtake
```

### Pit Stop Confounding Problem
**Issue:** Position changes due to pit stops are NOT on-track overtakes.

**Mitigation strategies:**
1. **Exclude pit laps:** Remove opportunities where either driver pits in [L, L+K]
2. **Compound change detection:** Filter out laps with tire compound changes
3. **Out-lap detection:** Exclude laps immediately after pit stops
4. **Outlier lap times:** Remove statistical outliers (formation laps, SC)
5. **Minimum stint length:** Require at least N clean laps (default: 3)

### Label Sensitivity Analysis
Compare labeling with different lookahead windows:
- **K=1:** Immediate next-lap overtake (stricter, lower prevalence)
- **K=2:** Overtake within 2 laps (more opportunities, noisier)
- **K=3:** Overtake within 3 laps (highest noise from pit confounding)

Recommended: **K=1** for cleaner labels despite lower sample size.

### Safety Car Detection
Laps with lap times > mean + 2σ flagged as potential SC laps and excluded.

## 3. Feature Engineering

### Gap Features
- **Gap (seconds):** Time gap to car ahead
- **Gap bins:** Categorical (very_close: <0.5s, close: 0.5-1.0s, medium: 1.0-2.0s, far: 2.0-3.0s)

### Pace Features
- **Relative pace:** Recent lap time difference (3-lap rolling average)
- **Pace ratio:** pace_behind / pace_ahead
- **Closing rate:** Change in gap over last 2 laps (temporal)

### Tire Features
- **Tire age:** Laps on current tires
- **Tire age difference:** age_behind - age_ahead (negative = advantage)
- **Compound advantage:** Ordinal encoding (SOFT=3, MEDIUM=2, HARD=1)
- **Same compound flag:** Binary indicator

### Temporal Features (Advanced)
- **Lagged gap:** Gap at L-1, L-2 laps
- **Lagged pace:** Pace delta at L-1, L-2
- **Trend features:** Slope of gap over last 3 laps

**Handling missing lags:** At stint start, use forward-fill or indicator variable.

### Race Phase Features
- **Lap number:** Absolute lap count
- **Race progress:** lap / total_laps (0 to 1)
- **Race phase:** Categorical (early: <0.33, middle: 0.33-0.66, late: >0.66)

### Track Features
- **Track identifier:** One-hot encoded circuit name
- **Track type:** Street circuit vs permanent (if available)

### Normalization
Continuous features standardized: `z = (x - μ) / σ` computed on training set only.

## 4. Train/Test Splitting

### Leakage-Safe Group Split
**Critical:** Split by race weekend, NOT randomly.

```
Races = [R1, R2, ..., RN]
Shuffle races randomly (with seed)
Train_races = first 70%
Test_races = last 30%
```

This prevents:
- Information leakage across laps in same race
- Overfitting to specific race conditions
- Unrealistic evaluation metrics

### Class Imbalance
- **Typical prevalence:** 2-5% positive class (overtakes are rare)
- **Handling:** Class weights + scale_pos_weight (XGBoost)

## 5. Model Training

### Baseline: Logistic Regression
```
P(y=1 | x) = σ(w^T x + b)
where σ(z) = 1 / (1 + e^(-z))
```
- Linear decision boundary
- Interpretable coefficients
- **Class weights:** Inverse of class frequency

### Advanced: XGBoost
```
F(x) = ∑ f_t(x)  for t=1 to T trees
P(y=1 | x) = σ(F(x))
```
- Gradient boosted decision trees
- Captures non-linear interactions
- **scale_pos_weight:** Ratio of negative to positive samples
- **Hyperparameters:** n_estimators, max_depth, learning_rate, subsample

### Optuna Hyperparameter Tuning
```
Objective: Maximize PR AUC (better for imbalanced data)
Search space:
  - n_estimators: [50, 500]
  - max_depth: [3, 10]
  - learning_rate: [0.01, 0.3]
  - subsample: [0.6, 1.0]
  - colsample_bytree: [0.6, 1.0]

Cross-validation: GroupKFold by race (k=5)
Early stopping: Patience = 20 rounds
Trials: 100
```

## 6. Probability Calibration

### Why Calibration Matters
Raw model scores may not reflect true probabilities.

**Example problem:**
- Model predicts 70% → Actual overtake rate is 50%
- Probabilities are poorly calibrated

### Calibration Methods

#### Isotonic Regression (Non-parametric)
```
f(s) = piecewise constant function fit to (scores, labels)
P_calibrated = f(P_raw)
```
- Flexible, fits arbitrary monotonic relationship
- Requires sufficient calibration data

#### Platt Scaling (Parametric)
```
P_calibrated = σ(a * logit(P_raw) + b)
```
- Assumes sigmoid-shaped miscalibration
- Works with less data

### Implementation
Use held-out calibration set (or cross-validation) to fit calibration function.

## 7. Evaluation Metrics

### Classification Metrics

#### ROC AUC (Receiver Operating Characteristic)
- Measures overall discrimination ability
- **Target:** >= 0.75 (good discrimination)

#### PR AUC (Precision-Recall)
- Better for imbalanced data
- **Target:** >= 2x positive prevalence

#### Brier Score
```
Brier = mean((P_predicted - y_actual)²)
```
- Measures calibration quality
- Lower is better
- **Target:** < 0.10

#### Confusion Matrix
At decision threshold (default 0.5):
- True Positives, False Positives
- True Negatives, False Negatives
- Precision, Recall, F1

### Calibration Metrics

#### Calibration Curve
Bin predicted probabilities, compare to actual frequencies.
- **Well-calibrated:** Points lie on diagonal
- **Overconfident:** Points below diagonal
- **Underconfident:** Points above diagonal

#### Expected Calibration Error (ECE)
```
ECE = ∑ (n_bin / n_total) * |accuracy_bin - confidence_bin|
```
- **Target:** < 0.05

## 8. Expected Overtakes (xO) Metric

### Definition
For a driver in a race:
```
xO = ∑ P(overtake | features) over all opportunities
```

### Interpretation
- **xO = 3.5:** Driver expected to make 3.5 overtakes given opportunities
- **Actual = 5:** Driver outperformed expected (skill, fortune, DRS)
- **Actual = 2:** Driver underperformed (traffic, errors, poor car)

### Leaderboard
Compare drivers by:
- xO (expected)
- Actual overtakes
- xO differential (actual - xO)

### Caveat
**xO is only meaningful if model is well-calibrated.**
- Check calibration curve before interpreting xO
- Large miscalibration → xO is biased

## 9. Model Ablation and Feature Importance

### Ablation Studies
Compare model variants:
1. **Static features only:** Gap + tire + race phase
2. **+ Temporal features:** Add lagged gap, closing rate
3. **+ Track features:** Add circuit identifier

**Metric:** PR AUC improvement

### Feature Importance
- **Permutation importance:** Shuffle feature, measure performance drop
- **XGBoost gain:** Total gain from splits on this feature
- **SHAP values (optional):** Individual prediction explanations

## 10. Limitations and Known Issues

### Label Noise
- **Pit detection imperfect:** Some pit-related position changes leak through
- **DRS not modeled:** Major overtaking factor ignored
- **Safety car periods:** Some may not be fully filtered
- **Incidents:** Mechanical failures, crashes cause position changes

### Missing Factors
- **DRS zones:** Not explicitly modeled (partially captured by track features)
- **Track overtaking difficulty:** Only partially captured by track ID
- **Traffic/backmarkers:** Multi-car interactions ignored
- **Weather:** Rain/wet conditions not handled
- **Driver skill:** Assumes all drivers equal

### Model Uncertainty
- **Confidence intervals:** Can be computed via bootstrap, but not by default
- **Out-of-distribution:** Model may fail on unusual scenarios (new tracks, rule changes)

### Interpretation Guidelines
Use xO for:
- Comparative analysis (driver performance vs expected)
- Scenario exploration (what-if sliders)
- Understanding overtaking factors

Do NOT use for:
- Absolute truth about individual overtakes
- Real-time race predictions (insufficient real-time data)
- Betting or commercial purposes

## 11. Reproducibility

### Deterministic Training
- All random seeds recorded in config
- Train/test split deterministic with seed
- Model training deterministic (with caveats for XGBoost multithreading)

### Dataset Versioning
Metadata file `datasets/metadata.json`:
```json
{
  "races": ["Bahrain 2024", "Saudi Arabia 2024", ...],
  "n_samples": 15240,
  "prevalence": 0.024,
  "build_timestamp": "2025-12-26T10:30:00",
  "code_version": "1.0.0"
}
```

### Outputs
```
outputs/<run_id>/
  ├── run_config.json       # All parameters
  ├── model_report.html     # Full evaluation report
  ├── metrics.json          # Numerical metrics
  └── figures/
      ├── pr_curve.png
      ├── roc_curve.png
      ├── calibration.png
      └── feature_importance.png
```

---

## References

1. **FastF1:** https://docs.fastf1.dev/
2. **scikit-learn:** https://scikit-learn.org/
3. **XGBoost:** https://xgboost.readthedocs.io/
4. **Probability Calibration:** Niculescu-Mizil & Caruana (2005)

---

**Document Version:** 1.0
**Last Updated:** 2025-12-26
**Author:** João Pedro Cunha
