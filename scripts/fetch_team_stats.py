import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from sqlmodel import Session

from voitto.database.database import engine
from voitto.database.models import TeamStats


def calculate_advanced_stats(season: str = "2023-24") -> pd.DataFrame:
    print(f"Fetching games for season {season}...")
    
    # 1. Fetch all games for the season
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",             # NBA
        season_type_nullable="Regular Season"
    )
    games = finder.get_data_frames()[0]
    
    # 2. Pre-process Dates and IDs
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values(["TEAM_ID", "GAME_DATE"])

    games["POSS"] = (
        games["FGA"] + 
        (0.44 * games["FTA"]) - 
        games["OREB"] + 
        games["TOV"]
    )

    # 3. Join with Opponent Data (Self-Join on Game ID)
    # We need opponent stats to calculate Pace and Def Rating
    games_merged = pd.merge(
        games, 
        games, 
        on="GAME_ID", 
        suffixes=("", "_OPP")
    )
    # Filter out the row where Team == Opponent
    games_merged = games_merged[
        games_merged["TEAM_ID"] != games_merged["TEAM_ID_OPP"]
    ]

    # 4. Feature Engineering
    
    # --- Context: Home/Away ---
    # "GSW vs. LAL" means Home; "GSW @ LAL" means Away
    games_merged["is_home"] = games_merged["MATCHUP"].str.contains(" vs. ")

    # --- Context: Rest Days ---
    # Calculate days since previous game for the specific team
    games_merged["prev_game_date"] = games_merged \
                                    .groupby("TEAM_ID")["GAME_DATE"].shift(1)
    games_merged["rest_days"] = pd.to_timedelta(
        games_merged["GAME_DATE"] - games_merged["prev_game_date"]
    ).dt.days - 1
    # Default to 3 for opener
    games_merged["rest_days"] = games_merged["rest_days"] \
                                .fillna(3).clip(lower=0) 

    # --- Advanced Stats Calculations ---
    
    # Pace: Possessions per 48 minutes
    # Note: MIN column is usually formatted as total minutes (e.g. 240)
    games_merged["pace"] = 48 * (
        (games_merged["POSS"] + games_merged["POSS_OPP"]) / 
        (2 * (games_merged["MIN"] / 5))
    )

    # Effective Field Goal %
    games_merged["efg_pct"] = (
        games_merged["FGM"] + 0.5 * games_merged["FG3M"]
    ) / games_merged["FGA"]

    # Turnover Rate (TOV per 100 possessions)
    games_merged["tov_pct"] = 100 * games_merged["TOV"] / games_merged["POSS"]

    # Offensive Rebound Rate
    # ORB / (ORB + Opponent DRB)
    games_merged["orb_pct"] = games_merged["OREB"] / (
        games_merged["OREB"] + games_merged["DREB_OPP"]
    )

    # Free Throw Rate
    games_merged["ft_rate"] = games_merged["FTM"] / games_merged["FGA"]

    # Defensive Rating
    # Points Allowed per 100 Possessions
    games_merged["def_rating"] = 100 * (
        games_merged["PTS_OPP"] / games_merged["POSS"]
    )

    return games_merged

def save_team_stats(df: pd.DataFrame) -> None:
    print(f"Saving {len(df)} team stat records...")
    with Session(engine) as session:
        for _, row in df.iterrows():
            stat = TeamStats(
                game_id=row["GAME_ID"],
                team_id=row["TEAM_ID"],
                game_date=row["GAME_DATE"],
                pace=row["pace"],
                efg_pct=row["efg_pct"],
                tov_pct=row["tov_pct"],
                orb_pct=row["orb_pct"],
                ft_rate=row["ft_rate"],
                def_rating=row["def_rating"],
                rest_days=int(row["rest_days"]),
                is_home=bool(row["is_home"])
            )
            session.add(stat)
        session.commit()
    print("Done!")

if __name__ == "__main__":
    stats_df = calculate_advanced_stats("2025-26")
    save_team_stats(stats_df)