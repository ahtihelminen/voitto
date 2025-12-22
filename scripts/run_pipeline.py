import subprocess
import time
from datetime import datetime

# --- Configuration ---
ODDS_TIME = "19:47"   # Time to fetch odds (Morning)
STATS_TIME = "19:49"  # Time to fetch stats (Next day/Early morning)

def run_script(script_name: str) -> None:
    """Executes a python script as a subprocess."""
    print(f"[{datetime.now()}] Starting {script_name}...")
    try:
        # Use uv run
        result = subprocess.run(
            ["uv", "run", script_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"[{datetime.now()}] {script_name} completed successfully.")
        else:
            print(
                f"[{datetime.now()}] Error running {script_name}:"
                f"\n{result.stderr}"
            )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[{datetime.now()}] Failed to run {script_name}: {e}")

def main() -> None:
    print(
        f"Pipeline scheduler started. "
        f"Odds at {ODDS_TIME}, Stats at {STATS_TIME}."
    )
    
    # Track if we have already run the task for the current minute
    # to prevent duplicate runs
    last_run_minute = None

    while True:
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # Check if we are in a new minute to avoid running
        # multiple times in the same minute
        if current_time != last_run_minute:
            
            if current_time == ODDS_TIME:
                run_script("src/voitto/ingest.py")
                last_run_minute = current_time
                
            elif current_time == STATS_TIME:
                run_script("src/voitto/fetch_stats.py")
                last_run_minute = current_time

        # Sleep for 30 seconds to spare CPU cycles
        time.sleep(30)

if __name__ == "__main__":
    main()