"""Streamlit app for F1 Overtake Probability Model.

Author: João Pedro Cunha
"""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from f1overtake.build_dataset import build_dataset
from f1overtake.calibrate import calibrate_all_models
from f1overtake.config import DEFAULT_CONFIG, QUICK_CONFIG
from f1overtake.evaluate import evaluate_all_models, get_calibration_data, get_feature_importance
from f1overtake.split import prepare_xy, split_by_race
from f1overtake.train import load_models, save_models, train_all_models
from f1overtake.viz import (
    create_calibration_plot,
    create_confusion_matrix_plot,
    create_feature_importance_plot,
    create_probability_heatmap,
    create_race_timeline_plot,
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="F1 Overtake Probability Model",
    page_icon="🏎️",
    layout="wide",
)


def main():
    """Main Streamlit app."""
    st.title("🏎️ F1 Overtake Probability Model")
    st.markdown("**Predict overtaking opportunities using machine learning**")
    st.markdown("*Author: João Pedro Cunha*")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    mode = st.sidebar.radio("Mode", ["Quick Demo (2 races)", "Full Analysis (24 races)"])
    config = QUICK_CONFIG if mode == "Quick Demo (2 races)" else DEFAULT_CONFIG

    # Show current configuration
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Setup")
    st.sidebar.markdown(f"**Races:** {len(config.data.race_names)} races")
    st.sidebar.markdown(f"**Expected Samples:** ~{len(config.data.race_names) * 650:,}")
    st.sidebar.markdown(f"**XGBoost Trees:** {config.model.xgb_n_estimators}")
    st.sidebar.markdown(f"**Year:** {config.data.year}")

    # Check if models exist
    models_dir = Path(config.data.models_dir)
    models_exist = models_dir.exists() and len(list(models_dir.glob("*.pkl"))) > 0

    # Force rebuild option
    force_rebuild = st.sidebar.checkbox("Force rebuild dataset", value=False, help="Rebuild dataset even if cached")

    if not models_exist:
        st.sidebar.warning("No trained models found. Click 'Build/Load Dataset' and 'Train Models'.")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dataset & Training", "🎯 Model Evaluation", "🔮 Predictions", "ℹ️ About"]
    )

    # Tab 1: Dataset & Training
    with tab1:
        st.header("Dataset & Training")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Build/Load Dataset"):
                with st.spinner("Building dataset..."):
                    dataset = build_dataset(config, force_rebuild=force_rebuild)
                    st.session_state["dataset"] = dataset
                    st.session_state["config_used"] = mode  # Track which config was used
                    st.success(f"Dataset loaded: {len(dataset)} samples")
                    if force_rebuild:
                        st.info("Dataset rebuilt from scratch")

        with col2:
            if st.button("Train Models"):
                if "dataset" not in st.session_state:
                    st.error("Please build dataset first")
                else:
                    with st.spinner("Training models..."):
                        dataset = st.session_state["dataset"]
                        train_df, test_df = split_by_race(dataset, test_size=0.3)

                        models = train_all_models(train_df, config)
                        calibrated = calibrate_all_models(models, test_df, config)

                        all_models = {**models, **calibrated}
                        save_models(all_models, config.data.models_dir)

                        st.session_state["models"] = all_models
                        st.session_state["train_df"] = train_df
                        st.session_state["test_df"] = test_df

                        st.success(f"Trained {len(all_models)} models")

        # Display dataset summary
        if "dataset" in st.session_state:
            dataset = st.session_state["dataset"]

            # Check if config changed
            if "config_used" in st.session_state and st.session_state["config_used"] != mode:
                st.warning(
                    f"⚠️ Dataset was built with {st.session_state['config_used']} but you're now in {mode}. "
                    "Check 'Force rebuild dataset' and click 'Build/Load Dataset' to rebuild with new configuration."
                )

            st.subheader("Dataset Summary")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(dataset))
            with col2:
                st.metric("Overtakes", int(dataset["Overtake"].sum()))
            with col3:
                st.metric("Overtake Rate", f"{dataset['Overtake'].mean()*100:.1f}%")
            with col4:
                st.metric("Races", dataset["RaceName"].nunique())

            # Show which races are in the dataset
            st.markdown(f"**Races in dataset:** {', '.join(dataset['RaceName'].unique())}")

    # Tab 2: Model Evaluation
    with tab2:
        st.header("Model Evaluation")

        # Load models if not in session
        if "models" not in st.session_state and models_exist:
            st.session_state["models"] = load_models(config.data.models_dir)
            if "dataset" not in st.session_state:
                st.session_state["dataset"] = build_dataset(config)
            dataset = st.session_state["dataset"]
            train_df, test_df = split_by_race(dataset, test_size=0.3)
            st.session_state["test_df"] = test_df

        if "models" in st.session_state and "test_df" in st.session_state:
            models = st.session_state["models"]
            test_df = st.session_state["test_df"]

            # Model selection
            model_name = st.selectbox("Select Model", list(models.keys()))
            model = models[model_name]

            # Evaluation metrics
            st.subheader("Performance Metrics")
            results = evaluate_all_models({model_name: model}, test_df)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("ROC AUC", f"{results.loc[model_name, 'roc_auc']:.3f}")
            with col2:
                st.metric("PR AUC", f"{results.loc[model_name, 'pr_auc']:.3f}")
            with col3:
                st.metric("Brier Score", f"{results.loc[model_name, 'brier_score']:.3f}")
            with col4:
                st.metric("Accuracy", f"{results.loc[model_name, 'accuracy']:.3f}")

            # Calibration plot
            st.subheader("Calibration Curve")
            cal_data = get_calibration_data(model, test_df)
            cal_fig = create_calibration_plot(cal_data, model_name)
            st.plotly_chart(cal_fig, use_container_width=True)

            # Feature importance
            st.subheader("Feature Importance")
            importance_df = get_feature_importance(model)
            if not importance_df.empty:
                imp_fig = create_feature_importance_plot(importance_df)
                st.plotly_chart(imp_fig, use_container_width=True)

            # Confusion matrix
            st.subheader("Confusion Matrix")
            metrics = results.loc[model_name].to_dict()
            cm_fig = create_confusion_matrix_plot(metrics, model_name)
            st.plotly_chart(cm_fig, use_container_width=True)

        else:
            st.info("Please build dataset and train models first")

    # Tab 3: Predictions
    with tab3:
        st.header("Overtake Probability Predictions")

        if "models" in st.session_state:
            models = st.session_state["models"]
            model_name = st.selectbox(
                "Select Model for Predictions", list(models.keys()), key="pred_model"
            )
            model = models[model_name]

            st.subheader("What-If Scenario Analysis")

            col1, col2, col3 = st.columns(3)
            with col1:
                gap = st.slider("Gap to Car Ahead (s)", 0.0, 3.0, 1.0, 0.1)
            with col2:
                tire_age_diff = st.slider("Tire Age Advantage (laps)", -20, 20, 0)
            with col3:
                relative_pace = st.slider("Relative Pace (s/lap)", -1.0, 1.0, 0.0, 0.1)

            # Create feature vector for prediction
            features = {
                "Gap": gap,
                "RelativePace": relative_pace,
                "PaceRatio": 1.0 - (relative_pace / 90.0) if gap > 0 else 1.0,
                "TyreLife": 10,
                "AheadTyreLife": 10 - tire_age_diff,
                "TireAgeDiff": tire_age_diff,
                "CompoundAdvantage": 0,
                "RaceProgress": 0.5,
                "Position": 10,
            }

            # Align with model features
            if hasattr(model, "feature_names_"):
                X = pd.DataFrame([features])
                X = X.reindex(columns=model.feature_names_, fill_value=0)

                prob = model.predict_proba(X)[0, 1]

                st.metric("Overtake Probability", f"{prob*100:.1f}%")

                # Probability heatmap
                st.subheader("Probability Heatmap (Gap vs Tire Age)")
                heatmap_fig = create_probability_heatmap(model)
                st.plotly_chart(heatmap_fig, use_container_width=True)

            # Race timeline
            if "test_df" in st.session_state:
                st.subheader("Race Timeline Analysis")
                test_df = st.session_state["test_df"]

                available_races = test_df["RaceName"].unique()
                selected_race = st.selectbox("Select Race", available_races)

                race_df = test_df[test_df["RaceName"] == selected_race]
                available_drivers = race_df["Driver"].unique()
                selected_driver = st.selectbox("Select Driver", available_drivers)

                timeline_fig = create_race_timeline_plot(race_df, selected_driver, model)
                st.plotly_chart(timeline_fig, use_container_width=True)

        else:
            st.info("Please build dataset and train models first")

    # Tab 4: About
    with tab4:
        st.header("About This Model")

        st.markdown(
            """
        ### Overview
        This machine learning model predicts the probability of an overtake occurring in F1 races
        based on various factors including gap, tire age, and relative pace.

        ### Methodology
        **Labeling Strategy:**
        - An overtake opportunity is created for each driver-pair on each lap
        - Label is positive (1) if the driver overtakes within the next lap
        - Pit stops are filtered out to reduce confounding

        **Features:**
        - Gap to car ahead (seconds)
        - Relative pace indicators
        - Tire age difference
        - Compound advantage
        - Race progress
        - Position

        **Models:**
        - Logistic Regression (baseline)
        - XGBoost (gradient boosting)
        - Calibrated versions for reliable probabilities

        ### Limitations
        ⚠️ This model has several known limitations:
        - Position changes may occur without on-track overtakes (pit strategy)
        - Label noise exists due to incomplete pit detection
        - Safety car periods may not be fully filtered
        - DRS and track-specific factors not explicitly modeled
        - Traffic and weather conditions not considered

        ### Data Source
        All data comes from **FastF1**, a free and open-source F1 data API.

        ### Author
        **João Pedro Cunha**

        ### License
        MIT License - Free to use and modify
        """
        )


if __name__ == "__main__":
    main()
