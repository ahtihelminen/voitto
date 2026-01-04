from pathlib import Path

import arviz as az
import bambi as bmb
import pandas as pd


# Ensure we use the shared logic for creating formulas/families
def get_model_config(model_type: str) -> tuple[str, str]:
    """Returns (formula, family) based on model type."""
    if model_type == "Gaussian Residual":
        return (
            "target ~ 1 + (1 | player_name) + (1 | opponent_team)",
            "gaussian"
        )
    if model_type == "Poisson Base":
        return (
            "target ~ market_line + (1 | player_name) + (1 | opponent_team)",
            "poisson"
        )
    msg = f"Unknown model type: {model_type}"
    raise ValueError(msg)

def train_base_model(
    training_data: pd.DataFrame,
    config: dict[str, str | float],
    save_dir: str = "saved_models",
) -> Path:
    """
    Trains a base model on historical data and saves the trace to disk.
    
    Args:
        training_data: DataFrame containing historical games (23-24, etc.)
        config: Dict containing 'model_type' and 'model_name'
        save_dir: Directory to save the .nc file
    
    Returns:
        str: Path to the saved NetCDF file.
    """
    model_type = str(config["model_type"])
    model_name = str(config.get("model_name", "model"))
    
    # 1. Prepare Data
    training_df = training_data.copy()
    formula, family = get_model_config(model_type)
    
    if model_type == "Gaussian Residual":
        training_df["target"] = (
            training_df["points"] - training_df["market_line"]
        )
    else:
        training_df["target"] = training_df["points"]

    print(f"🏗 Training Base Model ({model_type}) on {len(training_df)} rows...")

    # 2. Define & Fit
    model = bmb.Model(formula, data=training_df, family=family, dropna=True)
    
    # We use higher draws for the base model to ensure robust priors
    idata = model.fit(draws=1000, tune=1000, chains=2, progressbar=True)
    
    # 3. Save Trace
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / f"{model_name}_base.nc"
    
    # Overwrite if exists
    if save_path.exists():
        save_path.unlink()
        
    az.to_netcdf(idata, save_path)
    print(f"✅ Model saved to: {save_path}")
    
    return save_path