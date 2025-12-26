# scripts/backtest_engine.py
from __future__ import annotations

from collections.abc import Callable

import arviz as az
import bambi as bmb
import pandas as pd
from arviz import InferenceData


def extract_priors_from_posterior(
    inference_data: InferenceData,
    model_family: str,
    recency_weight: float = 1.0,
) -> dict[str, bmb.Prior]:
    """Extract informative priors from a fitted model's posterior."""
    posterior_summary = az.summary(inference_data, round_to=4, kind="stats")
    extracted_priors: dict[str, bmb.Prior] = {}

    # Extract priors for common model terms
    for term_name in ["Intercept", "market_line"]:
        if term_name in posterior_summary.index:
            prior_mean = float(
                posterior_summary.loc[term_name, "mean"] # type: ignore
            )
            prior_sd = (
                float(
                    posterior_summary.loc[term_name, "sd"] # type: ignore
                )
                * recency_weight
            )
            extracted_priors[term_name] = bmb.Prior(
                "Normal", mu=prior_mean, sigma=prior_sd
            )

    # Family specific params (Sigma for Gaussian)
    # We can't easily set a prior for sigma in Bambi's high-level API yet
    # without complex custom families, so we rely on the data for sigma.
    _ = model_family  # acknowledge unused in this block

    return extracted_priors


def run_backtest(
    player_stats_df: pd.DataFrame,
    backtest_config: dict[str, str | int | float],
    progress_callback: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """
    Run walk-forward backtest with specified configuration.

    backtest_config: dict with keys
        [start_date, model_type, recency_weight, retrain_days]
    """
    # 1. Feature Engineering based on Model Type
    is_residual_model = backtest_config["model_type"] == "Gaussian Residual"
    if is_residual_model:
        player_stats_df["target"] = (
            player_stats_df["points"] - player_stats_df["market_line"]
        )
        model_formula = "target ~ 1 + (1 | player_name) + (1 | opponent_team)"
        model_family = "gaussian"
    else:
        player_stats_df["target"] = player_stats_df["points"]
        model_formula = (
            "target ~ market_line + (1 | player_name) + (1 | opponent_team)"
        )
        model_family = "poisson"

    # 2. Split into training and test sets
    test_start_date = pd.to_datetime(
        backtest_config["test_start_date"], utc=True
    )
    training_df = player_stats_df[
        player_stats_df["game_date"] < test_start_date
    ].copy()
    test_df = player_stats_df[
        player_stats_df["game_date"] >= test_start_date
    ].copy()

    test_dates = test_df["game_date"].unique()
    num_test_days = len(test_dates)

    # 3. Initial model training to establish priors
    if progress_callback:
        progress_callback(0.0, "Training Base Model (Priors)...")

    base_model = bmb.Model(
        model_formula, data=training_df, family=model_family, dropna=True
    )
    base_inference_data = base_model.fit(
        draws=500, tune=500, chains=2, progressbar=False
    )

    assert isinstance(backtest_config["recency_weight"], float)
    learned_priors = extract_priors_from_posterior(
        base_inference_data, model_family, backtest_config["recency_weight"]
    )

    # 4. Walk-Forward Loop
    daily_predictions: list[pd.DataFrame] = []
    expanding_training_df = training_df.copy()
    current_model = base_model
    current_inference_data = base_inference_data
    last_retrain_date = None

    for day_index, current_date in enumerate(test_dates):
        # Update progress
        progress_pct = day_index / num_test_days
        if progress_callback:
            progress_callback(
                progress_pct,
                f"Processing {pd.to_datetime(current_date).date()}...",
            )

        # Check if model retraining is needed
        days_since_retrain = (
            (current_date - last_retrain_date).days
            if last_retrain_date
            else 999
        )
        assert isinstance(backtest_config["retrain_days"], int)
        if days_since_retrain >= backtest_config["retrain_days"]:
            retrained_model = bmb.Model(
                model_formula,
                data=expanding_training_df,
                priors=learned_priors,
                family=model_family,
                dropna=True,
            )
            current_inference_data = retrained_model.fit(
                draws=200, tune=200, chains=1, progressbar=False
            )
            current_model = retrained_model
            last_retrain_date = current_date

        # Generate predictions for current day
        current_day_df = test_df[test_df["game_date"] == current_date]
        current_model.predict(
            current_inference_data,
            data=current_day_df,
            kind="response",
            sample_new_groups=True,
        )

        predicted_values = (
            current_inference_data.posterior_predictive["target"]
            .mean(dim=["chain", "draw"])
            .values
        )

        # Convert residual predictions back to point predictions if needed
        predictions_df = current_day_df.copy()
        if is_residual_model:
            predictions_df["pred_points"] = (
                predictions_df["market_line"] + predicted_values
            )
        else:
            predictions_df["pred_points"] = predicted_values

        daily_predictions.append(predictions_df)
        expanding_training_df = pd.concat(
            [expanding_training_df, current_day_df]
        )

    results_df = pd.concat(daily_predictions)
    results_df["error"] = results_df["pred_points"] - results_df["points"]
    return results_df