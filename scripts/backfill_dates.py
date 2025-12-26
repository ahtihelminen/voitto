# backfill_dates.py
from sqlmodel import Session, create_engine, select, text

from voitto.models import GameOdds, Unified

# Update with your DB path
sqlite_url = "sqlite:///voitto.db"
engine = create_engine(sqlite_url)

def add_column_if_missing() -> None:
    with Session(engine) as session:
        try:
            # Check if column exists
            session.exec(text("SELECT game_date FROM unified LIMIT 1")) # type: ignore
        except Exception:  # noqa: BLE001
            print("Adding 'game_date' column to Unified table...")
            # We add it as a string/datetime column
            session.exec(
                text("ALTER TABLE unified ADD COLUMN game_date DATETIME") # type: ignore
            )
            session.commit()

def backfill_game_data() -> None:
    print("Starting date backfill...")
    with Session(engine) as session:
        # Join Unified with GameOdds to get the time
        statement = select(Unified, GameOdds).where(
            Unified.odds_game_id == GameOdds.id
        )
        results = session.exec(statement).all()
        
        count = 0
        for unified_row, odds_row in results:
            # Only update if missing
            if not unified_row.game_date:
                unified_row.game_date = odds_row.commence_time
                session.add(unified_row)
                count += 1
            
            if count % 1000 == 0 and count > 0:
                print(f"Updated {count} rows...")
        
        session.commit()
        print(f"✅ Success! Backfilled timestamps for {count} games.")

if __name__ == "__main__":
    add_column_if_missing()
    backfill_game_data()