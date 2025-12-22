import time

from sqlmodel import Session, create_engine

from voitto.stats_utils import (
    fetch_season_logs,
    save_nba_games_to_db,
    save_stats_to_db,
)

# --- Config ---
DATABASE_URL = "sqlite:///voitto.db"
SEASONS = ["2025-26","2024-25","2023-24"]

def main() -> None:
    engine = create_engine(DATABASE_URL)
    
    with Session(engine) as session:
        for season in SEASONS:
            print(f"Processing {season}...")
            # Let errors propagate if fetch fails
            season_stats = fetch_season_logs(season)
            save_nba_games_to_db(session, season_stats)
            save_stats_to_db(session, season_stats)
            
            # Polite API pause
            time.sleep(2) 

if __name__ == "__main__":
    main()