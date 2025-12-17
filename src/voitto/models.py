from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class Game(SQLModel, table=True):
    """Stores the schedule of games."""
    id: str = Field(primary_key=True)  # TheOddsAPI 'id' (e.g., "122e4dc...")
    sport_key: str                     # e.g., "basketball_nba"
    home_team: str
    away_team: str
    commence_time: datetime
    
class PlayerPropOdds(SQLModel, table=True):
    """
    The 'Ground Truth' lines from bookmakers.
    Example: LeBron James Over 24.5 Points @ 1.90
    """
    id: int | None = Field(default=None, primary_key=True)
    game_id: str = Field(foreign_key="game.id") # Links to Game table
    bookmaker: str       # e.g. "DraftKings"
    player_name: str     # e.g. "LeBron James"
    market_key: str      # e.g. "player_points"
    
    # The Line (The Target)
    point: float         # e.g. 24.5
    
    # The Prices
    odds_over: float     # e.g. 1.90
    odds_under: float    # e.g. 1.90
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)  # noqa: UP017
    )

class PlayerGameStats(SQLModel, table=True):
    """
    The 'Ground Truth' results from the game.
    Source: Scraped Box Scores (e.g. Basketball Reference)
    """
    id: int | None = Field(default=None, primary_key=True)
    source_game_id: str  # e.g. "202310230LAL"
    game_date: datetime # To help match with odds
    player_name: str    # The join key (needs to be normalized later)
    team: str
    
    # The Actuals
    points: int
    rebounds: int
    assists: int
    
    # Meta
    source: str = "basketball_reference"
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)  # noqa: UP017
    )

