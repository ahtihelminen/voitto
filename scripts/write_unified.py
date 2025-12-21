from datetime import timedelta

from sqlmodel import Session, create_engine, select

from voitto.models import (
    GameOdds,
    GameStats,
    PlayerPropOdds,
    PlayerStats,
    Unified,
)

DATABASE_URL = "sqlite:///voitto.db"

def normalize_name(name: str) -> str:
    """Normalize player names (remove punctuation, lower case)."""
    # e.g. "C.J. McCollum" -> "cj mccollum"
    return name.strip().lower().replace(".", "")

def main() -> None:
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        print("Loading data...")
        # Load all data into memory
        all_odds = session.exec(select(PlayerPropOdds)).all()
        game_odds_map = {
            game.id: game for game in session.exec(select(GameOdds)).all()
        }
        
        # Load Stats Schedule: Group by Date -> Home_Team -> GameStats
        # Structure: { "2023-12-01": { "LAL": GameStatsObj } }
        stats_schedule_map = {}
        all_games_stats = session.exec(select(GameStats)).all()
        
        for game_stats in all_games_stats:
            game_date = game_stats.game_date.date().isoformat()
            if game_date not in stats_schedule_map:
                stats_schedule_map[game_date] = {}
            # Map both Home/Away for lookup. 
            stats_schedule_map[game_date][game_stats.home_team] = game_stats
            stats_schedule_map[game_date][game_stats.away_team] = game_stats

        # Load Player Stats: (GameID, NormalizedName) -> PlayerStats
        player_stats_map = {}
        all_players_stats = session.exec(select(PlayerStats)).all()
        for player_stats in all_players_stats:
            key = (
                player_stats.source_game_id,
                normalize_name(player_stats.player_name)
            )
            player_stats_map[key] = player_stats

        unified_records = []
        matches = 0
        misses = 0

        print(f"Processing {len(all_odds)} odds entries...")

        for odds in all_odds:
            game_odds = game_odds_map.get(odds.game_id)
            if not game_odds:
                continue

            # --- SIMPLIFIED LOGIC START ---
            # 1. Team Match: Directly use the abbreviation from DB
            home_team = game_odds.home_team 

            # 2. Date Match: Handle UTC vs Local Time (+/- 1 Day)
            matched_game_stats = None
            date_candidates = [0, 1, -1] 
            
            for offset in date_candidates:
                # Convert UTC timestamp to Date + Offset
                check_date = (
                    game_odds.commence_time.date() + timedelta(days=offset)
                ).isoformat()
                
                # Check if that Date exists in our stats schedule 
                # and team played on that date
                if (
                    check_date in stats_schedule_map and 
                    home_team in stats_schedule_map[check_date]
                ):
                    matched_game_stats = stats_schedule_map[
                        check_date
                    ][
                        home_team
                    ]
                    break
            # --- SIMPLIFIED LOGIC END ---
            
            if not matched_game_stats:
                misses += 1
                continue

            # 3. Player Match
            p_key = (matched_game_stats.id, normalize_name(odds.player_name))
            p_stats = player_stats_map.get(p_key)

            if not p_stats:
                misses += 1
                continue

            # 4. Create Unified Record
            unified = Unified(
                odds_game_id=odds.game_id,
                stats_game_id=matched_game_stats.id,
                bookmaker=odds.bookmaker,
                player_name=odds.player_name,
                market_key=odds.market_key,
                market_line=odds.market_line,
                odds_over=odds.odds_over,
                odds_under=odds.odds_under,
                # Game Context
                wl=p_stats.wl,
                minutes=p_stats.minutes,
                plus_minus=p_stats.plus_minus,
                # Shooting
                fgm=p_stats.fgm,
                fga=p_stats.fga,
                fg_pct=p_stats.fg_pct,
                fg3m=p_stats.fg3m,
                fg3a=p_stats.fg3a,
                fg3_pct=p_stats.fg3_pct,
                ftm=p_stats.ftm,
                fta=p_stats.fta,
                ft_pct=p_stats.ft_pct,
                # Rebounding
                oreb=p_stats.oreb,
                dreb=p_stats.dreb,
                rebounds=p_stats.rebounds,
                # Playmaking & Turnover
                assists=p_stats.assists,
                turnovers=p_stats.turnovers,
                # Defense
                steals=p_stats.steals,
                blocks=p_stats.blocks,
                blka=p_stats.blka,
                # Fouls
                fouls=p_stats.fouls,
                fouls_drawn=p_stats.fouls_drawn,
                # Scoring & Misc
                points=p_stats.points,
                nba_fantasy_pts=p_stats.nba_fantasy_pts,
                dd2=p_stats.dd2,
                td3=p_stats.td3,
            )
            unified_records.append(unified)
            matches += 1

        print(
            f"Saving {len(unified_records)} matched records"
            f" (Missed: {misses})..."
        )
        
        # Clear old data (optional) and save new
        session.exec(select(Unified)).all() 
        session.add_all(unified_records)
        session.commit()

if __name__ == "__main__":
    main()