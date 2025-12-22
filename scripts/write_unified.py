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
    return name.strip().lower().replace(".", "")

def main() -> None:
    engine = create_engine(DATABASE_URL)

    with Session(engine) as session:
        print("Loading data...")
        all_odds = session.exec(select(PlayerPropOdds)).all()
        game_odds_map = {
            game.id: game for game in session.exec(select(GameOdds)).all()
        }
        
        # Load Stats Schedule: Group by Date -> Home_Team -> GameStats
        stats_schedule_map = {}
        all_games_stats = session.exec(select(GameStats)).all()
        
        for game_stats in all_games_stats:
            game_date = game_stats.game_date.date().isoformat()
            if game_date not in stats_schedule_map:
                stats_schedule_map[game_date] = []
            stats_schedule_map[game_date].append(game_stats)

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

            # --- MATCHING LOGIC ---
            odds_teams = {game_odds.home_team, game_odds.away_team}
            matched_game_stats = None
            
            # Check Date +/- 1 Day (to handle UTC vs Local shift)
            date_candidates = [0, 1, -1] 
            
            for offset in date_candidates:
                check_date = (
                    game_odds.commence_time.date() + timedelta(days=offset)
                ).isoformat()
                
                potential_games = stats_schedule_map.get(check_date, [])
                
                for stats_game in potential_games:
                    # STRICT MATCH: Both teams must match
                    stats_teams = {stats_game.home_team, stats_game.away_team}
                    
                    # Intersection check (handles Home/Away swap)
                    if odds_teams == stats_teams:
                        matched_game_stats = stats_game
                        break
                
                if matched_game_stats:
                    break
            
            if not matched_game_stats:
                misses += 1
                # Optional: Uncomment for noisy debugging
                # print(f"[MISS GAME] {game_odds.home_team} vs "
                #       f"{game_odds.away_team} on {game_odds.commence_time}")
                continue

            # 3. Player Match
            p_key = (matched_game_stats.id, normalize_name(odds.player_name))
            p_stats = player_stats_map.get(p_key)

            if not p_stats:
                misses += 1
                # Only print if we are sure the game is correct (it is now)
                # print(f"[MISS PLAYER] {odds.player_name} in "
                #       f"{matched_game_stats.matchup}")
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
                # Stats
                fgm=p_stats.fgm,
                fga=p_stats.fga,
                fg_pct=p_stats.fg_pct,
                fg3m=p_stats.fg3m,
                fg3a=p_stats.fg3a,
                fg3_pct=p_stats.fg3_pct,
                ftm=p_stats.ftm,
                fta=p_stats.fta,
                ft_pct=p_stats.ft_pct,
                oreb=p_stats.oreb,
                dreb=p_stats.dreb,
                rebounds=p_stats.rebounds,
                assists=p_stats.assists,
                turnovers=p_stats.turnovers,
                steals=p_stats.steals,
                blocks=p_stats.blocks,
                blka=p_stats.blka,
                fouls=p_stats.fouls,
                fouls_drawn=p_stats.fouls_drawn,
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
        
        # Clear old table content? (Optional, safe for now to append or you 
        # can truncate)
        # session.exec(delete(Unified)) 
        
        session.add_all(unified_records)
        session.commit()

if __name__ == "__main__":
    main()