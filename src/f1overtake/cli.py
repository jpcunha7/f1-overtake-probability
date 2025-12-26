"""Command-line interface for F1 Overtake Probability Model.

Author: João Pedro Cunha

This module provides CLI commands for:
- Training models
- Evaluating models
- Scoring races
- Generating reports
"""

import argparse
import logging
import sys
from pathlib import Path


from f1overtake.build_dataset import build_dataset
from f1overtake.calibrate import calibrate_all_models
from f1overtake.config import DEFAULT_CONFIG, QUICK_CONFIG
from f1overtake.evaluate import evaluate_all_models
from f1overtake.model_report import generate_html_report
from f1overtake.split import prepare_xy, split_by_race
from f1overtake.train import load_models, save_models, train_all_models
from f1overtake.xo_metric import create_xo_leaderboard, export_xo_table

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def train_command(args):
    """Train models command.

    Args:
        args: Command-line arguments
    """
    logger.info("Starting training...")

    # Select config
    cfg = QUICK_CONFIG if args.quick else DEFAULT_CONFIG

    # Build dataset
    logger.info("Building dataset...")
    dataset = build_dataset(cfg)

    # Split
    logger.info("Splitting dataset...")
    train_df, test_df = split_by_race(
        dataset, test_size=cfg.model.test_size, random_seed=cfg.model.random_seed
    )

    # Train models
    logger.info("Training models...")
    models = train_all_models(train_df, cfg, use_tuning=args.tune)

    # Calibrate models
    if args.calibrate:
        logger.info("Calibrating models...")
        calibrated = calibrate_all_models(models, test_df, cfg)
        models.update(calibrated)

    # Save models
    logger.info("Saving models...")
    save_models(models, cfg.data.models_dir)

    # Evaluate
    logger.info("Evaluating models...")
    results = evaluate_all_models(models, test_df)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(results.to_string())
    print("=" * 60)

    # Generate reports
    if args.report:
        logger.info("Generating HTML reports...")
        outputs_dir = Path(cfg.data.outputs_dir)
        for model_name, model in models.items():
            if "calibrated" not in model_name:
                report_path = outputs_dir / f"{model_name}_report.html"
                generate_html_report(
                    model, model_name, train_df, test_df, cfg, str(report_path)
                )

    logger.info("Training complete!")


def evaluate_command(args):
    """Evaluate models command.

    Args:
        args: Command-line arguments
    """
    logger.info("Starting evaluation...")

    # Select config
    cfg = QUICK_CONFIG if args.quick else DEFAULT_CONFIG

    # Load models
    logger.info("Loading models...")
    models = load_models(cfg.data.models_dir)

    if not models:
        logger.error("No models found. Please train models first.")
        return

    # Build dataset
    logger.info("Building dataset...")
    dataset = build_dataset(cfg)

    # Split
    logger.info("Splitting dataset...")
    train_df, test_df = split_by_race(
        dataset, test_size=cfg.model.test_size, random_seed=cfg.model.random_seed
    )

    # Evaluate
    logger.info("Evaluating models...")
    results = evaluate_all_models(models, test_df)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(results.to_string())
    print("=" * 60)

    # Generate reports if requested
    if args.report:
        logger.info("Generating HTML reports...")
        outputs_dir = Path(cfg.data.outputs_dir)
        for model_name, model in models.items():
            if "calibrated" not in model_name:
                report_path = outputs_dir / f"{model_name}_report.html"
                generate_html_report(
                    model, model_name, train_df, test_df, cfg, str(report_path)
                )


def score_race_command(args):
    """Score a specific race command.

    Args:
        args: Command-line arguments
    """
    logger.info(f"Scoring race: {args.event} {args.year}")

    # Select config
    cfg = QUICK_CONFIG if args.quick else DEFAULT_CONFIG

    # Load models
    logger.info("Loading models...")
    models = load_models(cfg.data.models_dir)

    if not models:
        logger.error("No models found. Please train models first.")
        return

    # Select model
    model_name = args.model or "xgboost_calibrated"
    if model_name not in models:
        logger.error(f"Model {model_name} not found. Available: {list(models.keys())}")
        return

    model = models[model_name]

    # Build dataset
    logger.info("Building dataset...")
    dataset = build_dataset(cfg)

    # Filter to specific race
    race_df = dataset[
        (dataset["RaceName"].str.contains(args.event, case=False))
    ].copy()

    if len(race_df) == 0:
        logger.error(f"No data found for race: {args.event}")
        return

    race_name = race_df["RaceName"].iloc[0]
    logger.info(f"Found race: {race_name}")

    # Filter to specific driver if provided
    if args.driver:
        race_df = race_df[race_df["Driver"] == args.driver.upper()].copy()
        if len(race_df) == 0:
            logger.error(f"No data found for driver: {args.driver}")
            return

    # Prepare features
    X, y = prepare_xy(race_df)
    if hasattr(model, "feature_names_"):
        X = X[model.feature_names_]

    # Get predictions
    probabilities = model.predict_proba(X)[:, 1]

    # Add to dataframe
    results = race_df[["LapNumber", "Driver", "DriverAhead", "Gap", "Overtake"]].copy()
    results["Probability"] = probabilities
    results = results.sort_values("Probability", ascending=False)

    # Print results
    print("\n" + "=" * 80)
    print(f"RACE SCORING: {race_name}")
    if args.driver:
        print(f"Driver: {args.driver.upper()}")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)

    # Calculate xO
    from f1overtake.xo_metric import calculate_xo

    xo_df = calculate_xo(race_df, model)
    xo_leaderboard = create_xo_leaderboard(xo_df, race_name=race_name)

    print("\n" + "=" * 80)
    print("xO LEADERBOARD (Expected Overtakes)")
    print("=" * 80)
    print(xo_leaderboard.to_string(index=False))
    print("=" * 80)

    # Save to file if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save predictions
        pred_path = output_path.parent / f"{output_path.stem}_predictions.csv"
        results.to_csv(pred_path, index=False)
        logger.info(f"Saved predictions to {pred_path}")

        # Save xO
        xo_path = output_path.parent / f"{output_path.stem}_xo.csv"
        export_xo_table(xo_df, str(xo_path), format="csv")
        logger.info(f"Saved xO to {xo_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="F1 Overtake Probability Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "--quick", action="store_true", help="Use quick mode (fewer races)"
    )
    train_parser.add_argument(
        "--tune", action="store_true", help="Use Optuna hyperparameter tuning"
    )
    train_parser.add_argument(
        "--calibrate", action="store_true", help="Calibrate models", default=True
    )
    train_parser.add_argument(
        "--report", action="store_true", help="Generate HTML reports", default=True
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument(
        "--quick", action="store_true", help="Use quick mode (fewer races)"
    )
    eval_parser.add_argument(
        "--report", action="store_true", help="Generate HTML reports"
    )

    # Score race command
    score_parser = subparsers.add_parser("score-race", help="Score a specific race")
    score_parser.add_argument(
        "--year", type=int, default=2024, help="Year of the race"
    )
    score_parser.add_argument(
        "--event", type=str, required=True, help="Event name (e.g., 'Bahrain')"
    )
    score_parser.add_argument(
        "--driver", type=str, help="Driver code (e.g., 'VER')"
    )
    score_parser.add_argument(
        "--model", type=str, help="Model to use (default: xgboost_calibrated)"
    )
    score_parser.add_argument(
        "--output", type=str, help="Output directory for results"
    )
    score_parser.add_argument(
        "--quick", action="store_true", help="Use quick mode (fewer races)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Dispatch command
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "score-race":
        score_race_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
