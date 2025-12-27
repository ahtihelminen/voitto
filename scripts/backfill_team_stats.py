import logging

from nba_api.stats.static import teams
from sqlmodel import Session, select, text

from voitto.database import engine

# Adjust these imports to match your project structure
from voitto.models import TeamStats, Unified

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_new_columns() -> None:
    """
    Manually adds the new feature columns to the existing 'unified' table.
    """
    print("Migrating schema...")
    
    # We use separate statements for maximum compatibility (SQLite/Postgres)
    statements = [
        "ALTER TABLE unified ADD COLUMN team_pace FLOAT",
        "ALTER TABLE unified ADD COLUMN team_efg_pct FLOAT",
        "ALTER TABLE unified ADD COLUMN team_rest_days INTEGER",
        "ALTER TABLE unified ADD COLUMN opp_def_rating FLOAT",
        "ALTER TABLE unified ADD COLUMN opp_pace FLOAT",
    ]

    with Session(engine) as session:
        for stmt in statements:
            try:
                session.exec(text(stmt)) # type: ignore
                print(f"Success: {stmt}")
            except Exception as _:  # noqa: BLE001
                # This catches errors if the column already exists
                print(f"Skipped (might already exist): {stmt}")
                # print(f"Reason: {e}") 
        
        session.commit()
    print("Schema migration complete.")


def get_team_id_map() -> dict[str, int]:
    """
    Creates a case-insensitive lookup dictionary to map:
    Full Name ("Boston Celtics") -> ID
    Nickname ("Celtics") -> ID
    Abbreviation ("BOS") -> ID
    City ("Boston") -> ID
    """
    nba_teams = teams.get_teams()
    team_map = {}
    
    for t in nba_teams:
        t_id = t['id']
        # Map various name formats to the single static ID
        team_map[t['full_name'].lower()] = t_id
        team_map[t['nickname'].lower()] = t_id
        team_map[t['abbreviation'].lower()] = t_id
        team_map[t['city'].lower()] = t_id
        
    return team_map


def backfill_unified_team_stats() -> None:
    """
    Iterates through Unified table and populates team/opponent advanced stats
    using data from the TeamStats table.
    """
    logger.info("Starting backfill for Unified team stats...")
    
    with Session(engine) as session:
        # 1. Load Data
        logger.info("Fetching TeamStats dictionary...")
        all_team_stats = session.exec(select(TeamStats)).all()
        
        # Create Lookup: (game_id, team_id) -> TeamStats Object
        # We ensure game_id is string to match Unified schema
        stats_lookup = {
            (str(s.game_id), s.team_id): s 
            for s in all_team_stats
        }

        logger.info("Fetching Unified records...")
        unified_records = session.exec(select(Unified)).all()
        
        # 2. Prepare Helpers
        team_map = get_team_id_map()
        updated_count = 0
        
        # 3. Iterate and Update
        for row in unified_records:
            # Skip if game_id is missing
            if not row.stats_game_id:
                continue

            # Resolve Team IDs
            try:
                # Handle potential case sensitivity or extra whitespace
                p_team_key = str(row.player_team).strip().lower()
                o_team_key = str(row.opponent_team).strip().lower()
                
                my_team_id = team_map.get(p_team_key)
                opp_team_id = team_map.get(o_team_key)
            except AttributeError:
                # Happens if team name is None
                continue

            if not my_team_id or not opp_team_id:
                # logger.warning(f"Could not map teams for row {row.id}: 
                # {row.player_team} vs {row.opponent_team}")
                continue

            # --- A. Update My Team Context ---
            # Lookup Key: (Game ID, My Team ID)
            my_stats = stats_lookup.get((str(row.stats_game_id), my_team_id))
            
            if my_stats:
                row.team_pace = my_stats.pace
                row.team_efg_pct = my_stats.efg_pct
                row.team_rest_days = my_stats.rest_days

            # --- B. Update Opponent Context ---
            # Lookup Key: (Game ID, Opponent Team ID)
            opp_stats = stats_lookup.get((str(row.stats_game_id), opp_team_id))
            
            if opp_stats:
                row.opp_def_rating = opp_stats.def_rating
                row.opp_pace = opp_stats.pace

            session.add(row)
            updated_count += 1
            
            # Optional: Batch commit if dataset is huge
            if updated_count % 1000 == 0:
                logger.info(f"Processed {updated_count} rows...")

        # 4. Final Commit
        logger.info(f"Committing updates for {updated_count} records...")
        session.commit()
        logger.info("Backfill complete.")

if __name__ == "__main__":
    add_new_columns()
    backfill_unified_team_stats()