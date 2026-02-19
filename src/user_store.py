import sqlite3
from typing import List
from datetime import datetime
import uuid


class UserStore:
    """
    Handles user state persistence using SQLite.
    This includes:
    - users
    - agent categories (collections) created by users
    """

    def __init__(self, db_path: str = "users.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create required tables if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

            conn.commit()

    # ---------- USER ----------

    def create_user(self) -> str:
        """
        Create a new user and return its ID.
        """
        user_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO users (id, created_at) VALUES (?, ?)",
                (user_id, created_at),
            )
            conn.commit()

        return user_id

    def get_or_create_user(self) -> str:
        """
        Returns an existing user if present, otherwise creates one.
        Assumes a single-user local environment.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT id FROM users LIMIT 1")
            row = cursor.fetchone()

            if row:
                return row[0]

        return self.create_user()

    # ---------- COLLECTIONS ----------

    def add_collection(self, user_id: str, name: str) -> str:
        """
        Add a new agent category (collection) for a user.
        Returns the collection ID.
        """
        collection_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO collections (id, user_id, name, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (collection_id, user_id, name, created_at),
            )
            conn.commit()

        return collection_id

    def get_user_collections(self, user_id: str) -> List[str]:
        """
        Return a list of collection names for a given user.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM collections WHERE user_id = ?",
                (user_id,),
            )
            return [row[0] for row in cursor.fetchall()]
