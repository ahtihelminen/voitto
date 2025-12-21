import sys

import voitto.models
from voitto.database import engine


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run scripts/init_table.py <ExactModelName>")
        sys.exit(1)
        
    model_name = sys.argv[1]
    
    # 1. Get the class directly (Case-Sensitive)
    try:
        model = getattr(voitto.models, model_name)
    except AttributeError:
        print(f"Error: Model '{model_name}' not found in voitto.models.")
        print("Check casing (e.g., 'PlayerStats', not 'playerstats').")
        sys.exit(1)

    print(f"Reinitializing table for: {model_name}...")

    # 2. Drop
    try:
        model.__table__.drop(engine)
        print("   -> Dropped table.")
    except Exception as e:  # noqa: BLE001
        print(f"   -> Warning (drop failed): {e}")

    # 3. Create
    try:
        model.__table__.create(engine)
        print("   -> Created table.")
    except Exception as e:  # noqa: BLE001
        print(f"!!! Error creating table: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()