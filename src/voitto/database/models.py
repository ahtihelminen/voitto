import json
from datetime import datetime, timezone
from typing import Any, ClassVar

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

    team_pace: float | None = Field(default=None)
    team_efg_pct: float | None = Field(default=None)
    team_rest_days: int | None = Field(default=None)

    opp_def_rating: float | None = Field(default=None)
    opp_pace: float | None = Field(default=None)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

class ModelArtifact(SQLModel, table=True):
    """
    Registry of trained model components.
    """
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True) 
    
    # Identity
    model_type: str      # e.g. "xgboost", "bambi_gaussian", "sklearn_ridge"
    target_feature: str  # e.g. "usg_pct", "true_shooting", "pace"
    
    # Storage
    artifact_path: str
    
    # Config & Metadata (Stored as JSON strings)
    hyperparameters: str = Field(default="{}")
    feature_cols: str = Field(default="[]")
    filters: str = Field(default="{}")

    metrics: str | None = Field(default=None)
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def config(self) -> dict[str, Any]:
        """Helper to get full config as dict"""
        return {
            "hyperparameters": json.loads(self.hyperparameters),
            "features": json.loads(self.feature_cols)
        }

class PredictionPipeline(SQLModel, table=True):
    """
    Defines a recipe for combining models (The Assembly).
    """
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    
    # References to ModelArtifact IDs
    # e.g. '{"usage_model_id": 12, "efficiency_model_id": 15, ...}'
    recipe_config: str 
    
    active: bool = Field(default=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

class DailyPrediction(SQLModel, table=True):
    """Stores the specific prediction for a specific game & model."""
    id: int | None = Field(default=None, primary_key=True)
    model_artifact_id: int = Field(foreign_key="modelartifact.id")
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

class TeamStats(SQLModel, table=True):
    """
    Stores advanced team metrics per game.
    Source: nba_api LeagueDashTeamStats & LeagueGameFinder
    """
    id: int | None = Field(default=None, primary_key=True)
    game_id: str = Field(index=True)
    team_id: int = Field(index=True)
    game_date: datetime
    
    # --- The Four Factors & Pace ---
    pace: float           # PACE
    efg_pct: float        # EFG_PCT
    tov_pct: float        # TM_TOV_PCT
    orb_pct: float        # OREB_PCT
    ft_rate: float        # FTA_RATE
    
    # --- Defense ---
    def_rating: float     # DEF_RATING
    
    # --- Context ---
    rest_days: int | None = Field(default=None) # Calculated from schedule
    is_home: bool = Field(default=False)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

