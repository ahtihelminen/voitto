from datetime import datetime, timedelta

from sqlmodel import Session, create_engine

from voitto.database.stats_utils import (
    fetch_season_logs,
    save_nba_games_to_db,
    save_stats_to_db,
)

# --- Config ---
DATABASE_URL = "sqlite:///voitto.db"
CURRENT_SEASON = "2025-26"

def main() -> None:
    engine = create_engine(DATABASE_URL)
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Running daily fetch for {yesterday}...")
    
    # 1. Fetch full current season
    season_stats = fetch_season_logs(CURRENT_SEASON)
    
    if season_stats.empty:
        print("Fetched data is empty.")
        return

    # 2. Filter to yesterday's games
    yesterday_stats = season_stats[
        season_stats["GAME_DATE"] == yesterday
    ].copy()
    
    if not yesterday_stats.empty:
        with Session(engine) as session:
            # A. Save Game Metadata (New Step)
            print("Saving Game Metadata...")
            save_nba_games_to_db(session, yesterday_stats)
            
            # B. Save Player Stats
            print("Saving Player Stats...")
            save_stats_to_db(session, yesterday_stats)
    else:
        print("No games found for yesterday.")

if __name__ == "__main__":
    main()