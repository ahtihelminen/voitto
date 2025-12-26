from datetime import datetime, timezone

import arviz as az
import bambi as bmb
import pandas as pd
from sqlmodel import Session, create_engine, select

from voitto.models import Unified

# --- Configuration ---
SQLITE_URL = "sqlite:///voitto.db"
MARKET_KEY = "player_points"
TEST_SEASON_START = datetime(2025, 10, 1, tzinfo=timezone.utc)
RECENCY_WEIGHT = 2.0
RETRAIN_INTERVAL_DAYS = 7  # Retrain only once per week

engine = create_engine(SQLITE_URL)

def get_data():
    """Fetches clean data sorted by date."""
    with Session(engine) as session:
        statement = select(Unified).where(
            Unified.market_key == MARKET_KEY,
            Unified.points is not None,
            Unified.game_date is not None,
        )
        results = session.exec(statement).all()
        df = pd.DataFrame([r.model_dump() for r in results])
        df['game_date'] = pd.to_datetime(df['game_date'], utc=True)
        return df.sort_values('game_date')

def extract_priors(idata, recency_weight=1.0):
    """Extracts posteriors from a trace to use as priors."""
    summary = az.summary(idata, round_to=4, kind="stats")
    
    int_mean = summary.loc["Intercept", "mean"]
    int_sd = summary.loc["Intercept", "sd"] * recency_weight
    
    slope_mean = summary.loc["market_line", "mean"]
    slope_sd = summary.loc["market_line", "sd"] * recency_weight
    
    return {
        "Intercept": bmb.Prior("Normal", mu=int_mean, sigma=int_sd),
        "market_line": bmb.Prior("Normal", mu=slope_mean, sigma=slope_sd),
    }

def run_backtest():
    print("--- 1. Loading & Splitting Data ---")
    df = get_data()
    
    df_history = df[df['game_date'] < TEST_SEASON_START].copy()
    df_test = df[df['game_date'] >= TEST_SEASON_START].copy()

    if df_test.empty:
        print("❌ No future data found for backtesting.")
        return

    print(f"History: {len(df_history)} games | Test Season: {len(df_test)} games")

    # --- PHASE 1: INITIAL BASELINE (Priors) ---
    print("\n--- 2. Learning Historical Priors (23-25) ---")
    model_base = bmb.Model(
        "points ~ market_line + (1 | player_name) + (1 | opponent_team)",
        data=df_history,
        family="poisson",
        dropna=True
    )
    # Fit initial priors
    idata_base = model_base.fit(draws=500, tune=500, chains=2, random_seed=42)
    current_priors = extract_priors(idata_base, RECENCY_WEIGHT)
    
    # Initialize "Active" Model state
    df_train_window = df_history.copy()
    active_model = None
    active_idata = None
    last_train_date = None
    
    results = []
    unique_dates = df_test['game_date'].unique()
    
    print(f"\n--- 3. Starting Walk-Forward Validation ({len(unique_dates)} Days) ---")

    # --- PHASE 2: DAILY LOOP ---
    for date in unique_dates:
        # Check if we need to retrain
        days_since = (date - last_train_date).days if last_train_date else 999
        
        if days_since >= RETRAIN_INTERVAL_DAYS:
            print(f" ⟳ Retraining on {date.date()} (Last: {days_since} days ago)...")
            
            # Create new model with updated window and priors
            new_model = bmb.Model(
                "points ~ market_line + (1 | player_name) + (1 | opponent_team)",
                data=df_train_window, 
                priors=current_priors,
                family="poisson", 
                dropna=True
            )
            
            new_idata = new_model.fit(draws=200, tune=200, chains=1, progressbar=False)
            
            active_model = new_model
            active_idata = new_idata
            last_train_date = date
        else:
            print(f" → Predicting {date.date()} (Cached Model)...")

        # Predict 'Today'
        df_today = df_test[df_test['game_date'] == date]
        
        active_model.predict(
            active_idata, 
            data=df_today, 
            kind="response", 
            sample_new_groups=True  # CRITICAL FIX for new players/teams
        )
        
        # Extract predictions
        pps = active_idata.posterior_predictive["points"].mean(dim=["chain", "draw"]).values
        
        df_results_today = df_today.copy()
        df_results_today["pred_points"] = pps
        results.append(df_results_today)
        
        # Update Window for next iteration (append today's data to history)
        df_train_window = pd.concat([df_train_window, df_today])

    # Final Stats
    final_df = pd.concat(results)
    final_df["error"] = final_df["pred_points"] - final_df["points"]
    mae = final_df['error'].abs().mean()
    
    print("\n--- Backtest Complete ---")
    print(f"MAE: {mae:.4f}")
    
    final_df.to_csv("backtest_results.csv", index=False)

if __name__ == "__main__":
    run_backtest()