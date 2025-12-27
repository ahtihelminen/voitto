import arviz as az
import bambi as bmb
import pandas as pd

# Re-use the extractor from your backtest_engine logic
from voitto.engine.train import get_model_config


def extract_priors_from_trace(
    trace_path: str, 
    recency_weight: float = 1.0
) -> dict[str, bmb.Prior]:
    """Loads a saved trace and extracts priors for the new model."""
    idata = az.from_netcdf(trace_path)
    summary = az.summary(idata, round_to=4, kind="stats")
    
    priors = {}
    # Extract fixed effects
    for term in ["Intercept", "market_line"]:
        if term in summary.index:
            mu = float(summary.loc[term, "mean"]) # type: ignore
            sd = float(summary.loc[term, "sd"]) * recency_weight # type: ignore
            priors[term] = bmb.Prior("Normal", mu=mu, sigma=sd)
            
    return priors

def predict_daily(
    current_season_data: pd.DataFrame,
    upcoming_games: pd.DataFrame,
    base_model_path: str,
    config: dict[str, str | float],
) -> pd.DataFrame:
    """
    Generates predictions for upcoming games using a Transfer Learning approach.
    
    1. Loads priors from 'base_model_path' (Historical knowledge).
    2. Retrains a model on 'current_season_data' (Recent knowledge).
    3. Predicts 'upcoming_games'.
    """
    model_type = str(config["model_type"])
    recency_weight = float(config.get("recency_weight", 1.0))
    
    # 1. Setup Data
    train_df = current_season_data.copy()
    predict_df = upcoming_games.copy()
    
    formula, family = get_model_config(model_type)
    
    if model_type == "Gaussian Residual":
        train_df["target"] = train_df["points"] - train_df["market_line"]
        # For prediction data, we don't have 'points' yet, target is placeholder
        predict_df["target"] = 0 
    else:
        train_df["target"] = train_df["points"]
        predict_df["target"] = 0

    # 2. Load Priors
    print(f"📥 Loading priors from {base_model_path}...")
    priors = extract_priors_from_trace(base_model_path, recency_weight)

    # 3. Fit Daily Model (The "Update" Step)
    # We fit on the current season using historical priors
    print(f"🔄 Fine-tuning on {len(train_df)} recent games...")
    model = bmb.Model(
        formula, 
        data=train_df, 
        priors=priors, 
        family=family, 
        dropna=True
    )
    
    # Faster fit for daily updates (fewer draws needed if priors are strong)
    idata = model.fit(draws=500, tune=500, chains=1, progressbar=False)

    # 4. Predict
    print(f"🔮 Predicting {len(predict_df)} upcoming games...")
    model.predict(
        idata, 
        data=predict_df, 
        kind="response", 
        sample_new_groups=True
    )

    # 5. Extract Results & Probabilities
    # Shape: (n_observations, n_chains * n_draws)
    posterior_samples = idata \
        .posterior_predictive["target"] \
        .stack(sample=("chain", "draw")) \
        .values
    
    results = predict_df.copy()
    
    if model_type == "Gaussian Residual":
        # Mean Prediction
        pred_diffs = posterior_samples.mean(axis=1)
        results["pred_diff"] = pred_diffs
        results["pred_points"] = results["market_line"] + pred_diffs
        
        # Probability Over (Residual > 0)
        # We assume the line is "fair" if residual is 0. 
        # If residual > 0, score > line.
        results["prob_over"] = (posterior_samples > 0).mean(axis=1)
        
    else:
        # Poisson Case
        pred_points = posterior_samples.mean(axis=1)
        results["pred_points"] = pred_points
        
        # Probability Over (Points > Market Line)
        # Broadcast market_line to shape (n_obs, 1) to compare with samples
        market_lines = results["market_line"].to_numpy()[:, None]
        results["prob_over"] = (posterior_samples > market_lines).mean(axis=1)

    return results