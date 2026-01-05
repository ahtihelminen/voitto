import os
import time
from datetime import datetime, timedelta, timezone  # <--- FIXED IMPORT
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlmodel import Session, select

from voitto.database.database import engine
from voitto.database.models import GameOdds, PlayerPropOdds
from voitto.database.teams import NAME_TO_ABBREVIATION_DICT

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")
SPORT = "basketball_nba"
START_DATE = "2024-01-07"  # API limit for player prop history
CSV_FILE = "historical_odds_backup.csv"

# --- Helpers ---
def get_historical_request(url: str, params: dict) -> dict | None:
    """Wrapper for requests to handle rate limits and errors."""
    params["api_key"] = API_KEY
    try:
        response = requests.get(url, params=params, timeout=15)
        
        # Rate Limit handling (Historical calls are expensive)
        remaining = int(response.headers.get("x-requests-remaining", 100))
        if remaining < 50:
            print(f"   [Quota Warning] Low quota: {remaining}. Sleeping 5s...")
            time.sleep(5)

        if response.status_code != 200:
            print(f"   Error {response.status_code}: {response.text}")
            return None
            
        return response.json()
    except Exception as e:  # noqa: BLE001
        print(f"   Request failed: {e}")
        return None

def main() -> None:
    # 1. Date Loop for Game Discovery
    print(f"--- Starting Historical Ingestion from {START_DATE} ---")
    current_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.now() - timedelta(days=1) 
    
    unique_game_ids = set()
    games_metadata = {} # id -> event_object

    print("1. Discovering Games (Daily Scans)...")
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%dT12:00:00Z") # Scan at noon
        print(f"   Scanning {current_date.strftime('%Y-%m-%d')}...", end="\r")
        
        response_json = get_historical_request(
            f"https://api.the-odds-api.com/v4/historical/sports/{SPORT}/events",
            {"date": date_str}
        )
        
        # Robust Data Extraction
        if response_json and "data" in response_json:
            events = response_json["data"]
            if isinstance(events, list):
                for event in events:
                    g_id = event["id"]
                    if g_id not in unique_game_ids:
                        unique_game_ids.add(g_id)
                        games_metadata[g_id] = event
        
        current_date += timedelta(days=1)
        time.sleep(0.3) 

    print(f"\nFound {len(unique_game_ids)} unique games.")

    # 2. Fetch Closing Lines for each Game
    print("\n2. Fetching Closing Lines (Player Props)...")
    
    # Prepare CSV headers if file doesn't exist
    csv_headers = ["game_id", "bookmaker", "player", "market", "line",
                    "over", "under", "timestamp"]
    if not Path(CSV_FILE).exists():
        pd.DataFrame(columns=csv_headers).to_csv(CSV_FILE, index=False)

    with Session(engine) as session:
        game_ids_list = list(unique_game_ids)
        
        for idx, game_id in enumerate(game_ids_list):
            event = games_metadata[game_id]
            commence_time = event["commence_time"]
            
            # --- FIX: Use explicit 'timezone.utc' ---
            commence_dt = datetime.fromisoformat(
                commence_time.replace("Z", "+00:00")  # noqa: FURB162
            )
            if commence_dt > datetime.now(timezone.utc):
                continue
            # ----------------------------------------

            print(
                f"   Processing {idx+1}/{len(game_ids_list)}:"
                f" {event['home_team']} vs {event['away_team']}"
            )

            # A. Ensure Game exists in DB
            db_game = session.get(GameOdds, game_id)
            if not db_game:
                try:
                    home_abbr = NAME_TO_ABBREVIATION_DICT[event["home_team"]]
                    away_abbr = NAME_TO_ABBREVIATION_DICT[event["away_team"]]
                    
                    new_game = GameOdds(
                        id=game_id,
                        sport_key=event["sport_key"],
                        home_team=home_abbr,
                        away_team=away_abbr,
                        commence_time=commence_dt
                    )
                    session.add(new_game)
                    session.commit()
                except Exception as e:  # noqa: BLE001
                    print(f"     Skipping game DB save (Data error): {e}")
                    continue

            # B. Fetch Odds at Commence Time (Closing Line)
            response_json = get_historical_request(
                f"https://api.the-odds-api.com/v4/historical/sports/{SPORT}/events/{game_id}/odds",
                {
                    "date": commence_time, 
                    "regions": "eu",
                    "markets": "player_points,player_rebounds,player_assists",
                    "bookmakers": "pinnacle",
                    "oddsFormat": "decimal"
                }
            )

            if not response_json or "data" not in response_json:
                continue

            # --- ROBUST DATA UNWRAPPING ---
            raw_data = response_json["data"]
            odds_snapshot = None

            if isinstance(raw_data, list):
                if raw_data:
                    odds_snapshot = raw_data[0]
                else:
                    continue
            elif isinstance(raw_data, dict):
                odds_snapshot = raw_data
            else:
                continue

            if "bookmakers" not in odds_snapshot:
                continue
            # -------------------------------

            # C. Process Odds
            rows_to_save = []
            db_objects = []
            
            ts_str = odds_snapshot.get(
                "timestamp",
                response_json.get("timestamp")
            )
            if ts_str:
                timestamp = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00")  # noqa: FURB162
                )
            else:
                timestamp = datetime.now(timezone.utc)

            for bm in odds_snapshot.get("bookmakers", []):
                for market in bm["markets"]:
                    # Group Overs and Unders
                    outcomes = {}
                    for out in market["outcomes"]:
                        key = (out["description"], out["point"])
                        if key not in outcomes:
                            outcomes[key] = {}
                        outcomes[key][out["name"]] = out["price"]
                    
                    for (player, line), prices in outcomes.items():
                        if "Over" in prices and "Under" in prices:
                            # 1. Add to CSV list
                            rows_to_save.append({
                                "game_id": game_id,
                                "bookmaker": bm["title"],
                                "player": player,
                                "market": market["key"],
                                "line": line,
                                "over": prices["Over"],
                                "under": prices["Under"],
                                "timestamp": ts_str
                            })
                            
                            # 2. Add to DB Object list
                            exists = session.exec(
                                select(PlayerPropOdds).where(
                                    PlayerPropOdds.game_id == game_id,
                                    PlayerPropOdds.player_name == player,
                                    PlayerPropOdds.market_key == market["key"],
                                    PlayerPropOdds.market_line == line,
                                    PlayerPropOdds.timestamp == timestamp
                                )
                            ).first()
                            
                            if not exists:
                                db_objects.append(PlayerPropOdds(
                                    game_id=game_id,
                                    bookmaker=bm["title"],
                                    player_name=player,
                                    market_key=market["key"],
                                    market_line=line,
                                    odds_over=prices["Over"],
                                    odds_under=prices["Under"],
                                    timestamp=timestamp
                                ))

            # D. Batch Save
            if rows_to_save:
                # CSV
                pd.DataFrame(rows_to_save).to_csv(CSV_FILE, mode='a',
                                                   header=False, index=False)
                # DB
                if db_objects:
                    session.add_all(db_objects)
                    session.commit()
                print(f"     Saved {len(rows_to_save)} props.")
            
            time.sleep(0.05) 

if __name__ == "__main__":
    main()