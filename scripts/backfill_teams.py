# backfill_teams.py
from sqlmodel import Session, create_engine, select, text

from voitto.database.models import GameStats, PlayerStats, Unified

# Update with your actual DB path
sqlite_url = "sqlite:///voitto.db"
engine = create_engine(sqlite_url)

def add_columns_if_missing() -> None:
    """Manually adds columns to SQLite if they don't exist."""
    with Session(engine) as session:
        try:
            # Try to query the new column to see if it exists
            session.exec(text("SELECT player_team FROM unified LIMIT 1")) # type: ignore
        except Exception:  # noqa: BLE001
            print("Adding 'player_team' column...")
            session.exec(
                text("ALTER TABLE unified ADD COLUMN player_team VARCHAR") # type: ignore
            )
            session.commit()

        try:
            session.exec(text("SELECT opponent_team FROM unified LIMIT 1")) # type: ignore
        except Exception:  # noqa: BLE001
            print("Adding 'opponent_team' column...")
            session.exec(
                text("ALTER TABLE unified ADD COLUMN opponent_team VARCHAR") # type: ignore
            )
            session.commit()

def backfill_data() -> None:
    print("Starting backfill...")
    with Session(engine) as session:
        # Fetch all unified rows
        unified_rows = session.exec(select(Unified)).all()
        
        count = 0
        for row in unified_rows:
            # 1. Find the matching PlayerStats to get the player's team
            # We match on Name and Game ID
            p_stat = session.exec(
                select(PlayerStats).where(
                    PlayerStats.player_name == row.player_name,
                    PlayerStats.source_game_id == row.stats_game_id
                )
            ).first()
            
            # 2. Find the GameStats to get the matchup details
            g_stat = session.exec(
                select(GameStats).where(GameStats.id == row.stats_game_id)
            ).first()
            
            if p_stat and g_stat:
                # Update Player Team
                row.player_team = p_stat.team
                
                # Update Opponent Team
                # If player is Home, Opponent is Away. Else Opponent is Home.
                if p_stat.team == g_stat.home_team:
                    row.opponent_team = g_stat.away_team
                else:
                    row.opponent_team = g_stat.home_team
                
                session.add(row)
                count += 1
                
                if count % 100 == 0:
                    print(f"Processed {count} rows...")
        
        session.commit()
        print(f"✅ Completed! Updated {count} rows.")

if __name__ == "__main__":
    add_columns_if_missing()
    backfill_data()