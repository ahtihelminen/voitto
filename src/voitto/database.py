from collections.abc import Generator
from typing import Any

from sqlmodel import Session, SQLModel, create_engine

# This creates a file named 'voitto.db' in the same directory
sqlite_file_name = "voitto.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

# check_same_thread=False is needed for SQLite when using FastAPI later
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo=False, connect_args=connect_args)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, Any, None]:
    with Session(engine) as session:
        yield session

if __name__ == "__main__":
    create_db_and_tables()
    print(f"Database initialized: {sqlite_file_name}")