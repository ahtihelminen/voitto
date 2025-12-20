from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import playergamelogs
from sqlmodel import Session, select

from voitto.models import GameStats, PlayerStats


def fetch_season_logs(season_str: str) -> pd.DataFrame:
    """
    Fetches raw game logs for a specific season from NBA API.
    """
    logs = playergamelogs.PlayerGameLogs(
        season_nullable=season_str,
        season_type_nullable='Regular Season'
    )
    season_stats = logs.get_data_frames()[0]
    # Normalize Date Format immediately
    season_stats['GAME_DATE'] = pd.to_datetime(
        season_stats['GAME_DATE']
    ).dt.strftime('%Y-%m-%d')
    return season_stats


def save_stats_to_db(session: Session, stats: pd.DataFrame) -> None:
    """
    Saves a DataFrame of player logs to the DB, skipping duplicates.
    Efficiently checks existing records in batch.
    """
    if stats.empty:
        return

    # 1. Pre-fetch existing IDs to avoid row-by-row DB queries
    # We use a composite key of (Source Game ID, Player Name)
    existing_query = select(
        PlayerStats.source_game_id,
        PlayerStats.player_name
    )
    existing_records = set(session.exec(existing_query).all())

    new_records = []
    
    for _, row in stats.iterrows():
        # Check if (GameID, PlayerName) tuple exists in our set
        if (str(row['GAME_ID']), row['PLAYER_NAME']) in existing_records:
            continue

        # Create Object
        stats_record = PlayerStats(
            source_game_id=str(row['GAME_ID']),
            game_date=datetime.strptime(row['GAME_DATE'], "%Y-%m-%d"),
            player_name=row['PLAYER_NAME'],
            team=row['TEAM_ABBREVIATION'],
            points=row['PTS'],
            rebounds=row['REB'],
            assists=row['AST'],
            steals=row['STL'],
            blocks=row['BLK'],
            threes=row['FG3M'],
            turnovers=row['TOV'],
            minutes=float(row['MIN'])
        )
        new_records.append(stats_record)

    # 2. Bulk Save
    if new_records:
        session.add_all(new_records)
        session.commit()
        print(f"   -> Committed {len(new_records)} new rows.")
    else:
        print("   -> No new unique records found.")


def save_nba_games_to_db(
        session: Session,
        stats: pd.DataFrame,
) -> None:
    """
    Extracts unique games from player stats and saves them to GameStats table.
    """
    if stats.empty:
        return

    # 1. Deduplicate to get one row per game
    # NBA API Matchups: "LAL @ BOS" (Away @ Home) or "BOS vs LAL" (Home vs Away)
    unique_games = stats.drop_duplicates(subset=["GAME_ID"]).copy()
    
    for _, row in unique_games.iterrows():
        game_id = str(row["GAME_ID"])
        matchup = row["MATCHUP"]
        
        # Parse Home/Away
        if " vs. " in matchup:
            # Format: "Home vs. Away"
            home, away = matchup.split(" vs. ")
        elif " @ " in matchup:
            # Format: "Away @ Home"
            away, home = matchup.split(" @ ")
        else:
            print(f"   !! Unknown matchup format: {matchup}")
            continue

        # Check existence
        if session.get(GameStats, game_id):
            continue

        game = GameStats(
            id=game_id,
            sport_key="basketball_nba",
            game_date=datetime.strptime(row["GAME_DATE"], "%Y-%m-%d"),
            home_team=home,
            away_team=away,
        )
        session.add(game)
    
    session.commit()
    print(f"   -> Processed game metadata for {len(unique_games)} games.")
    






