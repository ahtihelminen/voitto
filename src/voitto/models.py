from datetime import datetime

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
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PlayerGameStats(SQLModel, table=True):
    """
    The 'Features' & 'Labels' for training.
    We will scrape this later from a stats site.
    """
    id: int | None = Field(default=None, primary_key=True)
    game_id: str = Field(foreign_key="game.id")
    player_name: str
    team: str
    
    # The raw stats (Labels)
    points: int
    rebounds: int
    assists: int
    threes_made: int
    minutes: str         # e.g. "34:12"
    
    # We can add more advanced features (Usage %, PER) here later