import time

from sqlmodel import Session, create_engine

from voitto.stats_utils import fetch_season_logs, save_stats_to_db

# --- Config ---
DATABASE_URL = "sqlite:///voitto.db"
SEASONS = ["2023-24", "2022-23", "2021-22"]

def main() -> None:
    engine = create_engine(DATABASE_URL)
    
    with Session(engine) as session:
        for season in SEASONS:
            print(f"Processing {season}...")
            # Let errors propagate if fetch fails
            season_stats = fetch_season_logs(season)
            save_stats_to_db(session, season_stats)
            
            # Polite API pause
            time.sleep(2) 

if __name__ == "__main__":
    main()