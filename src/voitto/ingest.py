import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException
from sqlmodel import Session, desc, select

from voitto.database import create_db_and_tables, engine
from voitto.models import GameOdds, PlayerPropOdds
from voitto.teams import NAME_TO_ABBREVIATION_DICT

# 1. Setup
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")
SPORT = "basketball_nba"
REGIONS = "eu"
MARKETS = "player_points,player_rebounds,player_assists"
BOOKERS = "pinnacle"

def get_request(url: str, params: dict) -> dict | None:
    """Helper to handle requests with error checking."""
    try:
        response = requests.get(url, params=params, timeout=10)

        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")
        print(f"   [API Quota] Used: {used} | Remaining: {remaining}")

        # Handle Quota/Auth errors (401) or Throttling (429)
        if response.status_code in [401, 429]:
            print(f"!!! API Limit/Auth Error {response.status_code}:",
                  {response.reason})
            return None
            
        response.raise_for_status() # Raise for 4xx/5xx
        return response.json()
        
    except RequestException as exception:
        print(f"!!! Network/Request Error: {exception}")
        return None

def ingest_data() -> None:
    with Session(engine) as session:
        # --- Phase 1: Get Schedule ---
        print(f"1. Fetching Schedule for {SPORT}...")
        events = get_request(
            f"https://api.the-odds-api.com/v4/sports/{SPORT}/events",
            {"api_key": API_KEY}
        )
        
        if not events or "message" in events:
            print("   Failed to retrieve schedule.")
            return

        print(f"   Found {len(events)} games.")

        # Save Games to DB
        games_to_process = []
        for event in events:
            existing_game = session.get(GameOdds, event["id"])
            if not existing_game:
                game = GameOdds(
                    id=event["id"],
                    sport_key=event["sport_key"],
                    home_team=NAME_TO_ABBREVIATION_DICT[event["home_team"]],
                    away_team=NAME_TO_ABBREVIATION_DICT[event["away_team"]],
                    commence_time=datetime.fromisoformat(
                        event["commence_time"].replace("Z", "+00:00")  # noqa: FURB162
                    ),
                )
                session.add(game)
            games_to_process.append(event["id"])
        
        session.commit()

        # --- Phase 2: Get Player Props ---
        print(
            f"\n 2. Fetching Player Props for "
            f"{len(games_to_process)} games..."
        )
        
        for i, game_id in enumerate(games_to_process):
            print(f"   Processing Game {i+1}/{len(games_to_process)} "
                  f"(ID: {game_id})")

            data = get_request(
                f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{game_id}/odds",
                {
                    "api_key": API_KEY,
                    "regions": REGIONS,
                    "markets": MARKETS,
                    "bookmakers": BOOKERS,
                    "oddsFormat": "decimal",
                }
            )

            if not data:
                continue

            new_odds_count = 0
            for bookmaker in data.get("bookmakers", []):
                bm_name = bookmaker["title"]
                
                for market in bookmaker["markets"]:
                    market_key = market["key"]
                    
                    # Group outcomes by (Player, Point) to merge Over/Under
                    # Key: (desc, point) -> {'Over': price, 'Under': price}
                    grouped_outcomes = {}
                    
                    for outcome in market["outcomes"]:
                        player = outcome["description"]
                        market_line = outcome.get("point")
                        label = outcome["name"] # "Over" or "Under"
                        price = outcome.get("price")
                        
                        if (player, market_line) not in grouped_outcomes:
                            grouped_outcomes[(player, market_line)] = {
                                "Over": 0,
                                "Under": 0
                            }
                        
                        if label in ["Over", "Under"]:
                            grouped_outcomes[
                                (player, market_line)
                            ][label] = price

                    # Create DB records from grouped data
                    for (
                        player,
                        market_line
                    ), prices in grouped_outcomes.items():
                        # CHECK: Fetch latest odds for this specific line
                        statement = (
                            select(PlayerPropOdds)
                            .where(PlayerPropOdds.game_id == game_id)
                            .where(PlayerPropOdds.bookmaker == bm_name)
                            .where(PlayerPropOdds.player_name == player)
                            .where(PlayerPropOdds.market_key == market_key)
                            .where(PlayerPropOdds.market_line == market_line)
                            .order_by(desc(PlayerPropOdds.timestamp))
                            .limit(1)
                        )
                        latest_entry = session.exec(statement).first()

                        # Only add if prices changed or no history exists
                        if (not latest_entry or 
                            latest_entry.odds_over != prices["Over"] or 
                            latest_entry.odds_under != prices["Under"]):
                            
                            prop = PlayerPropOdds(
                                game_id=game_id,
                                bookmaker=bm_name,
                                player_name=player,
                                market_key=market_key,
                                market_line=market_line,
                                odds_over=prices["Over"],
                                odds_under=prices["Under"],
                            )
                            session.add(prop)
                            new_odds_count += 1

            try:
                session.commit()
                print(f"     Saved {new_odds_count} odds records.")
            except Exception as exception:  # noqa: BLE001
                print(f"     DB Error on commit: {exception}")
                session.rollback()

            time.sleep(1)  # Rate limiting
            
if __name__ == "__main__":
    create_db_and_tables()
    ingest_data()