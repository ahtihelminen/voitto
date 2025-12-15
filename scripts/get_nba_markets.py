import os  # noqa: INP001

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")

SPORT = "basketball_nba"
REGIONS = "us"
MARKETS = "player_points,player_rebounds,player_assists"

def check_markets():
    # --- Step 1: Get the Schedule (Event IDs) ---
    print("1️⃣  Fetching upcoming NBA games...")
    events_response = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{SPORT}/events",
        params={"api_key": API_KEY}
    )
    
    events = events_response.json()
    if not events or "message" in events:
        print("❌ No events found or error:", events)
        return

    # Pick the first game to test
    game = events[0]
    game_id = game["id"]
    print(f"   Found match: {game['home_team']} vs {game['away_team']} (ID: {game_id})")

    # --- Step 2: Get Player Props for ONE Game ---
    print(f"\n2️⃣  Fetching player props for this game...")
    odds_response = requests.get(
        f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{game_id}/odds", # <--- Note the URL change
        params={
            "api_key": API_KEY,
            "regions": REGIONS,
            "markets": MARKETS, 
            "oddsFormat": "decimal",
            "bookmakers": "fanduel,draftkings" 
        }
    )

    data = odds_response.json()
    
    # Check if we got data
    if "bookmakers" not in data:
        print("❌ Error fetching odds:", data)
        return

    # Scan for our target markets
    found_markets = set()
    for bookmaker in data["bookmakers"]:
        for market in bookmaker["markets"]:
            found_markets.add(market["key"])
            
    print("\n🎯 Result:")
    requested_list = MARKETS.split(",")
    for m in requested_list:
        if m in found_markets:
            print(f"   [✓] {m}")
        else:
            print(f"   [ ] {m} (Not found in this game)")

if __name__ == "__main__":
    check_markets()