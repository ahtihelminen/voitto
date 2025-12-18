import time

import pandas as pd
from nba_api.stats.endpoints import playergamelogs

# --- Configuration ---
SEASONS = ["2024-25", "2023-24", "2022-23", "2021-22"]  # Add more as needed
OUTPUT_FILE = ".data/nba_historical_player_stats.csv"


def fetch_season_stats(season_str: str) -> pd.DataFrame:
    """Fetch all player game logs for a given season."""
    print(f"Fetching stats for season: {season_str}...")
    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season_str,
        season_type_nullable="Regular Season",
    )
    return logs.get_data_frames()[0]
    
def main() -> None:
    """Backfill NBA player statistics from multiple seasons."""
    all_data = []

    for season in SEASONS:
        season_stats = fetch_season_stats(season)
        if not season_stats.empty:
            print(f"  - Retrieved {len(season_stats)} rows.")
            all_data.append(season_stats)
        # Sleep to be polite to the API
        time.sleep(2)

    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        # Filter for key columns relevant to props
        cols_to_keep = [
            "GAME_DATE",
            "MATCHUP",
            "PLAYER_NAME",
            "TEAM_ABBREVIATION",
            "PTS",
            "REB",
            "AST",
            "STL",
            "BLK",
            "FG3M",
            "TOV",
            "MIN",
        ]
        master_df = master_df[cols_to_keep]

        master_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Saved {len(master_df)} rows to {OUTPUT_FILE}")
    else:
        print("No data fetched.")


if __name__ == "__main__":
    main()