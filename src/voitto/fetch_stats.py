import time
import traceback
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup, Comment, Tag
from sqlmodel import Session, select

from voitto.database import create_db_and_tables, engine
from voitto.models import PlayerGameStats

BASE_URL = "https://www.basketball-reference.com/boxscores/"


def _get_stat_value(row_element: Tag, stat_name: str) -> int:
    """Extract integer stat value from a table row element."""
    cell = row_element.find("td", {"data-stat": stat_name})
    return int(cell.text) if cell and cell.text else 0

# CRITICAL: BRef blocks scripts without a real User-Agent
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def fetch_daily_stats(target_date: datetime) -> None:
    params = {
        "month": target_date.month,
        "day": target_date.day,
        "year": target_date.year
    }
    
    print(f"1. Fetching schedule for {target_date.date()}...")
    try:
        resp = requests.get(
            BASE_URL, params=params, headers=HEADERS, timeout=15
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"   Error fetching index page: {e}")
        return

    soup = BeautifulSoup(resp.content, "html.parser")
    
    box_links = []
    for link in soup.find_all("a", href=True):
        if "/boxscores/20" in link["href"] and ".html" in link["href"]:
            full_url = f"https://www.basketball-reference.com{link['href']}"
            if full_url not in box_links:
                box_links.append(full_url)
    
    print(f"   Found {len(box_links)} games. Scraping details...")

    new_stats_count = 0
    
    with Session(engine) as session:
        for url in box_links:
            game_id = url.split('/')[-1].replace('.html', '')
            print(f"   -> Processing: {game_id}")
            time.sleep(3) # INCREASED DELAY: BRef is very strict (3s is safer)
            
            try:
                game_resp = requests.get(url, headers=HEADERS, timeout=15)
                game_soup = BeautifulSoup(game_resp.content, "html.parser")
                
                # Extract tables hidden in comments
                comments = game_soup.find_all(
                    string=lambda text: isinstance(text, Comment)
                )
                all_soups = [game_soup]
                all_soups.extend([
                    BeautifulSoup(c, "html.parser")
                    for c in comments
                    if "<table" in c
                ])
                
                tables = []
                for s in all_soups:
                    # Find tables with IDs containing "game-basic"
                    all_tables = s.find_all("table")
                    for table in all_tables:
                        table_id = table.get("id")
                        if table_id and "game-basic" in table_id:
                            tables.append(table)

                if not tables:
                    print(
                        f"      [!] No box score tables found. "
                        f"(Status: {game_resp.status_code})"
                    )
                    # Check if we got a bot block page
                    if "The owner of this website" in game_resp.text:
                        print(
                            "      [!!!] BLOCKED by firewall. "
                            "Use a different IP or wait."
                        )
                    continue

                rows_processed = 0
                for table in tables:
                    # Identify team from table ID
                    # (e.g., "box-BOS-game-basic")
                    table_id = table.get("id", "")
                    team_code = (
                        table_id.split("-")[1] if "-" in table_id else "UNK"
                    )

                    rows = table.find("tbody").find_all("tr")
                    for row in rows:
                        if "thead" in row.get("class", []):
                            continue
                        
                        player_link = row.find("a", href=True)
                        if not player_link:
                            continue 
                            
                        name = player_link.text
                        rows_processed += 1
                        
                        # Extract stats
                        # (Don't use try/except here so we see errors)
                        mp_cell = row.find("td", {"data-stat": "mp"})
                        if not mp_cell or "Did Not" in mp_cell.text:
                            continue

                        pts = _get_stat_value(row, "pts")
                        trb = _get_stat_value(row, "trb")
                        ast = _get_stat_value(row, "ast")
                        
                        existing = session.exec(
                            select(PlayerGameStats)
                            .where(PlayerGameStats.source_game_id == game_id)
                            .where(PlayerGameStats.player_name == name)
                        ).first()
                        
                        if not existing:
                            stat = PlayerGameStats(
                                source_game_id=game_id,
                                game_date=target_date,
                                player_name=name,
                                team=team_code, 
                                points=pts,
                                rebounds=trb,
                                assists=ast
                            )
                            session.add(stat)
                            new_stats_count += 1
                
                print(f"      Parsed {rows_processed} rows.")

            except (requests.RequestException, ValueError) as e:
                print(f"      Failed to parse game {game_id}: {e}")
                traceback.print_exc()  # Show full error trace

        session.commit()
        print(f"Done. Saved {new_stats_count} records.")

if __name__ == "__main__":
    create_db_and_tables()
    yesterday = datetime.now() - timedelta(days=1)
    fetch_daily_stats(yesterday)