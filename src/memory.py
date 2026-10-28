"""
src/memory.py - Asterline Research Assistant
Persistent User-Scoped Memory Store using SQLite.

Provides persistent storage for user preferences and context across
application restarts, enforcing strict user-level isolation.

CRITICAL INVARIANT:
Personal memories represent user context/preferences ONLY.
Personal memories NEVER override official Asterline documentation or company policies.
"""

from contextlib import contextmanager
import os
import sqlite3
from typing import Dict, Generator, Optional

# Default database location within the project repository
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "memory.db",
)


class SQLiteMemoryStore:
    """
    Persistent SQLite-backed key-value store for user personal context and preferences.
    
    Features:
      - Survives process and application restarts.
      - Enforces strict user isolation via parameterized queries on `user_id`.
      - Automatically creates tables and directories if missing.
      - Safe connection lifecycle (explicitly closes connections to prevent file locks).
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initializes the memory store.
        
        Args:
          db_path: Path to the SQLite database file. Defaults to data/memory.db.
        """
        self.db_path = os.path.abspath(db_path or DEFAULT_DB_PATH)
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # Initialize schema
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager that yields a connection and ensures it is safely
        committed and closed, avoiding lingering file locks on Windows.
        """
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Creates the memories table and indexes if they do not already exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, key)
                );
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_id
                ON memories (user_id);
                """
            )
            conn.commit()

    def get_memories(self, user_id: str) -> Dict[str, str]:
        """
        Returns all stored memories for a given user as a {key: value} dictionary.
        Strictly isolated to user_id.
        """
        clean_user = user_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT key, value
                FROM memories
                WHERE user_id = ?
                ORDER BY key ASC;
                """,
                (clean_user,),
            )
            rows = cursor.fetchall()
            return {row["key"]: row["value"] for row in rows}

    def get_memory(self, user_id: str, key: str) -> Optional[str]:
        """
        Retrieves a single memory value for a user by key (case-insensitive key match).
        Returns the value string, or None if not found.
        """
        clean_user = user_id.strip()
        clean_key = " ".join(key.strip().split())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT value
                FROM memories
                WHERE user_id = ? AND key = ? COLLATE NOCASE
                LIMIT 1;
                """,
                (clean_user, clean_key),
            )
            row = cursor.fetchone()
            return row["value"] if row else None

    def set_memory(self, user_id: str, key: str, value: str) -> None:
        """
        Creates or updates a memory key-value pair for a user.
        If the key already exists (case-insensitive), updates its value and timestamp.
        """
        clean_user = user_id.strip()
        clean_key = " ".join(key.strip().split())
        clean_val = value.strip()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Check for existing key with case-insensitive match
            cursor.execute(
                """
                SELECT key
                FROM memories
                WHERE user_id = ? AND key = ? COLLATE NOCASE
                LIMIT 1;
                """,
                (clean_user, clean_key),
            )
            existing_row = cursor.fetchone()

            if existing_row:
                # Update existing record using its canonical existing key
                canonical_key = existing_row["key"]
                cursor.execute(
                    """
                    UPDATE memories
                    SET value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND key = ?;
                    """,
                    (clean_val, clean_user, canonical_key),
                )
            else:
                # Insert new record
                cursor.execute(
                    """
                    INSERT INTO memories (user_id, key, value, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP);
                    """,
                    (clean_user, clean_key, clean_val),
                )
            conn.commit()

    def delete_memory(self, user_id: str, key: str) -> bool:
        """
        Deletes a specific memory for a user (case-insensitive key match).
        Returns True if a record was deleted, or False if it was not found.
        """
        clean_user = user_id.strip()
        clean_key = " ".join(key.strip().split())
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM memories
                WHERE user_id = ? AND key = ? COLLATE NOCASE;
                """,
                (clean_user, clean_key),
            )
            conn.commit()
            return cursor.rowcount > 0

    def clear_memories(self, user_id: str) -> None:
        """
        Deletes all memories belonging to the specified user.
        Does not touch memories of any other user.
        """
        clean_user = user_id.strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM memories
                WHERE user_id = ?;
                """,
                (clean_user,),
            )
            conn.commit()

    def reset_all(self) -> None:
        """
        Clears all memories across all users. Used for test isolation.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM memories;")
            conn.commit()

