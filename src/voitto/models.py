from datetime import datetime, timezone
from typing import ClassVar

from sqlmodel import Field, SQLModel


class GameOdds(SQLModel, table=True):
    """Stores the schedule of games."""
    __tablename__: ClassVar[str] = "game_odds"  # type: ignore[misc]
    
    id: str = Field(primary_key=True)  # TheOddsAPI 'id' (e.g., "122e4dc...")
    sport_key: str                     # e.g., "basketball_nba"
    home_team: str
    away_team: str
    commence_time: datetime
    
class GameStats(SQLModel, table=True):
    """Stores the schedule of games."""
    __tablename__: ClassVar[str] = "game_stats"  # type: ignore[misc]
    
    id: str = Field(primary_key=True)  # TheOddsAPI 'id' (e.g., "122e4dc...")
    sport_key: str                     # e.g., "basketball_nba"
    home_team: str
    away_team: str
    game_date: datetime

class PlayerPropOdds(SQLModel, table=True):
    """
    The 'Ground Truth' lines from bookmakers.
    Example: LeBron James Over 24.5 Points @ 1.90
    """
    id: int | None = Field(default=None, primary_key=True)
    game_id: str = Field(foreign_key="game_odds.id") # Links to GameOdds table
    bookmaker: str       # e.g. "DraftKings"
    player_name: str     # e.g. "LeBron James"
    market_key: str      # e.g. "player_points"
    
    market_line: float         # e.g. 24.5
    
    # The Prices
    odds_over: float     # e.g. 1.90
    odds_under: float    # e.g. 1.90
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

class PlayerStats(SQLModel, table=True):
    """Full Box Score from NBA API."""
    id: int | None = Field(default=None, primary_key=True)
    source_game_id: str = Field(foreign_key="game_stats.id")
    game_date: datetime 
    player_name: str    
    team: str
    
    # --- Game Context ---
    wl: str | None     # "W" or "L"
    minutes: float     # "MIN"
    plus_minus: int    # "PLUS_MINUS"
    
    # --- Shooting ---
    fgm: int; fga: int; fg_pct: float  # noqa: E702
    fg3m: int; fg3a: int; fg3_pct: float  # noqa: E702
    ftm: int; fta: int; ft_pct: float  # noqa: E702
    
    # --- Rebounding ---
    oreb: int; dreb: int; rebounds: int  # noqa: E702
    
    # --- Playmaking & Turnover ---
    assists: int
    turnovers: int
    
    # --- Defense ---
    steals: int
    blocks: int
    blka: int          # "BLKA" (Blocked Attempts - shots blocked by opponent)
    
    # --- Fouls ---
    fouls: int         # "PF"
    fouls_drawn: int   # "PFD"
    
    # --- Scoring & Misc ---
    points: int
    nba_fantasy_pts: float
    dd2: int           # Double-Doubles
    td3: int           # Triple-Doubles

    source: str = "nba_api"
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Unified(SQLModel, table=True):
    """
    A unified view combining PlayerPropOdds and PlayerStats.
    Used for model training and evaluation.
    """
    id: int | None = Field(default=None, primary_key=True)
    odds_game_id: str = Field(foreign_key="game_odds.id")
    stats_game_id: str = Field(foreign_key="game_stats.id")
    game_date: datetime | None = Field(default=None)
    bookmaker: str
    player_name: str
    market_key: str
    market_line: float
    odds_over: float
    odds_under: float
    
     # --- Game Context ---
    wl: str | None     # "W" or "L"
    minutes: float     # "MIN"
    plus_minus: int    # "PLUS_MINUS"
    
    # --- Shooting ---
    fgm: int; fga: int; fg_pct: float  # noqa: E702
    fg3m: int; fg3a: int; fg3_pct: float  # noqa: E702
    ftm: int; fta: int; ft_pct: float  # noqa: E702
    
    # --- Rebounding ---
    oreb: int; dreb: int; rebounds: int  # noqa: E702
    
    # --- Playmaking & Turnover ---
    assists: int
    turnovers: int
    
    # --- Defense ---
    steals: int
    blocks: int
    blka: int          # "BLKA" (Blocked Attempts - shots blocked by opponent)
    
    # --- Fouls ---
    fouls: int         # "PF"
    fouls_drawn: int   # "PFD"
    
    # --- Scoring & Misc ---
    points: int
    nba_fantasy_pts: float
    dd2: int           # Double-Doubles
    td3: int           # Triple-Doubles

    player_team: str | None = Field(default=None)
    opponent_team: str | None = Field(default=None)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

class Experiment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True) # e.g. "Gaussian_Residual_v1"
    model_type: str # "gaussian_residual", "poisson_base"
    recency_weight: float
    training_cutoff: datetime
    description: str = Field(default="")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # We store the path to the "Base Priors" trace file
    base_model_path: str # e.g. "saved_models/exp_1_base.nc"

class DailyPrediction(SQLModel, table=True):
    """Stores the specific prediction for a specific game & model."""
    id: int | None = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="experiment.id")
    player_name: str
    game_date: datetime
    
    market_line: float
    predicted_diff: float # The raw model output
    final_prediction: float # The calculated point total
    prob_over: float # Probability for Kelly Criterion
    
    # Did we bet on it? (Populated later by the simulator)
    bet_placed: str | None # "Over", "Under", "Pass"
    bet_amount: float | None
    outcome: float | None # Profit/Loss

