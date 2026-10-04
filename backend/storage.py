from contextlib import contextmanager
from pathlib import Path
import json
import re
import sqlite3
import uuid

DATABASE = Path(__file__).resolve().parents[1] / "data/singularity_canvas.sqlite3"


def initialize_database():
    """Create missing tables on a fresh install without replacing saved data."""
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with connection() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                revision INTEGER NOT NULL DEFAULT 0,
                clues_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS sampling_runs (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                session_revision INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                model_sha256 TEXT NOT NULL,
                backend TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS explanations (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                session_revision INTEGER NOT NULL,
                run_id INTEGER NOT NULL REFERENCES sampling_runs(id),
                model TEXT NOT NULL,
                question TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                answer TEXT NOT NULL,
                elapsed_ms REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
        """)


class SessionConflict(Exception):
    """Another request changed this session."""


@contextmanager
def connection():
    db = sqlite3.connect(DATABASE, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    try:
        with db:
            yield db
    finally:
        db.close()


def normalize_username(value):
    username = value.strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,39}", username):
        raise ValueError(
            "Use 1–40 letters, numbers, underscores or hyphens; "
            "start with a letter or number."
        )
    return username


def validate_clues(clues):
    if not isinstance(clues, list) or len(clues) > 256:
        raise ValueError("Expected at most 256 clues.")
    seen = set()
    for clue in clues:
        if not isinstance(clue, dict) or set(clue) != {"index", "value"}:
            raise ValueError("Each clue needs an index and value.")
        index, value = clue["index"], clue["value"]
        if type(index) is not int or not 0 <= index < 256:
            raise ValueError("Invalid cell index.")
        if type(value) is not int or value not in (0, 1):
            raise ValueError("Invalid cell value.")
        if index in seen:
            raise ValueError("Duplicate cell index.")
        seen.add(index)
    return sorted(clues, key=lambda clue: clue["index"])


def session_data(db, session_id):
    row = db.execute(
        """
        SELECT s.*, u.username FROM sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        raise KeyError("Session not found.")
    result = dict(row)
    result["clues"] = json.loads(result.pop("clues_json"))
    run = db.execute(
        """
        SELECT id, result_json FROM sampling_runs
        WHERE session_id = ? AND session_revision = ?
        ORDER BY id DESC LIMIT 1
        """,
        (session_id, result["revision"]),
    ).fetchone()
    result["latest_batch"] = json.loads(run["result_json"]) if run else None
    return result


def open_session(username, new=False):
    username = normalize_username(username)
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "INSERT OR IGNORE INTO users(username) VALUES (?)", (username,)
        )
        user_id = db.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()["id"]
        row = None if new else db.execute(
            """
            SELECT id FROM sessions WHERE user_id = ?
            ORDER BY julianday(updated_at) DESC, rowid DESC LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        session_id = row["id"] if row else str(uuid.uuid4())
        if row is None:
            db.execute(
                """INSERT INTO sessions(id, user_id, created_at, updated_at)
                VALUES (?, ?,
                    strftime('%Y-%m-%d %H:%M:%f', 'now'),
                    strftime('%Y-%m-%d %H:%M:%f', 'now'))""",
                (session_id, user_id),
            )
        return session_data(db, session_id)


def get_session(session_id):
    with connection() as db:
        return session_data(db, session_id)


def save_clues(session_id, expected_revision, clues):
    clues = validate_clues(clues)
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("Invalid session revision.")
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        current = session_data(db, session_id)
        if current["revision"] != expected_revision:
            raise SessionConflict("Session changed; reload before saving.")
        if current["clues"] == clues:
            return current
        db.execute(
            """
            UPDATE sessions
            SET clues_json = ?, revision = revision + 1,
                updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')
            WHERE id = ?
            """,
            (json.dumps(clues), session_id),
        )
        return session_data(db, session_id)
