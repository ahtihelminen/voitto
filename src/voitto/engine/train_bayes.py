from pathlib import Path

import arviz as az
import bambi as bmb
import pandas as pd


def get_model_config(model_type: str, target_col: str) -> tuple[str, str]:
    """
    Returns (formula, family) based on model type and target.
    """
    # 1. Gaussian Residual (Legacy Betting Model)
    if model_type == "Gaussian Residual" and target_col == "points":
        return (
            "target ~ 1 + (1 | player_name) + (1 | opponent_team)",
            "gaussian",
        )

    # 2. Generic Gaussian (For Usage, Minutes, etc.)
    # If the user selected 'Gaussian Residual' but target is NOT points,
    # we treat it as a standard Normal model on the raw value.
    if "Gaussian" in model_type:
        return (f"{target_col} ~ 1 + (1 | player_name)", "gaussian")

    # 3. Poisson (For Counts like Assists, Rebounds, or raw Points)
    if model_type == "Poisson Base":
        # If we have a market line, use it as a base regressor?
        # For simplicity, we stick to intercept + player effects for components.
        return (
            f"{target_col} ~ 1 + (1 | player_name) + (1 | opponent_team)",
            "poisson",
        )

    msg = f"Unknown model config: {model_type} for target {target_col}"
    raise ValueError(msg)


def train_base_model(
    training_data: pd.DataFrame,
    config: dict[str, str | float],
    save_dir: str = "saved_models",
) -> Path:
    """
    Trains a Bayesian model component.
    """
    model_type = str(config["model_type"])
    experiment_name = str(config.get("experiment_name", "experiment"))
    target = str(config.get("target", "points"))

    # 1. Prepare Data
    training_df = training_data.copy()

    # Determine Formula
    formula, family = get_model_config(model_type, target)

    # Special Prep for Residuals
    # If predicting points residuals, we construct the 'target' column
    if model_type == "Gaussian Residual" and target == "points":
        if "market_line" not in training_df.columns:
            msg = "Gaussian Residual (Points) requires 'market_line' column."
            raise ValueError(
                msg
            )

        training_df["target"] = (
            training_df["points"] - training_df["market_line"]
        )
    else:
        # Ensure target exists
        if target not in training_df.columns:
            msg = f"Target column '{target}' not found in training data."
            raise ValueError(
                msg
            )

    print(
        f"Training ({model_type}) on target '{target}' with formula: {formula}"
    )

    # 2. Define & Fit
    model = bmb.Model(formula, data=training_df, family=family, dropna=True)

    # Fit
    idata = model.fit(draws=1000, tune=500, chains=2, progressbar=True)

    # 3. Save Trace
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f"{experiment_name}_base.nc"

    if save_path.exists():
        save_path.unlink()

    az.to_netcdf(idata, save_path)
    print(f"✅ Model saved to: {save_path}")

    return save_path
