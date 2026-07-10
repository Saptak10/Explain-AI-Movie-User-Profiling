import asyncio
import sqlite3
from pathlib import Path

_DB_PATH: str = ""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


async def execute(sql: str, params: tuple = ()) -> int:
    def _run():
        conn = _conn()
        cur = conn.execute(sql, params)
        conn.commit()
        rowid = cur.lastrowid
        conn.close()
        return rowid

    return await asyncio.to_thread(_run)


async def fetchone(sql: str, params: tuple = ()):
    def _run():
        conn = _conn()
        row = conn.execute(sql, params).fetchone()
        conn.close()
        return dict(row) if row else None

    return await asyncio.to_thread(_run)


async def fetchall(sql: str, params: tuple = ()):
    def _run():
        conn = _conn()
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    return await asyncio.to_thread(_run)


def init_db(path: str) -> None:
    global _DB_PATH
    _DB_PATH = path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            username         TEXT UNIQUE NOT NULL,
            hashed_password  TEXT NOT NULL,
            has_edited       INTEGER NOT NULL DEFAULT 0,
            sus_done         INTEGER NOT NULL DEFAULT 0,
            version          TEXT    NOT NULL DEFAULT 'O'
        )
    """)
    user_cols_before = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
    for col, defn in [
        ("has_edited", "INTEGER NOT NULL DEFAULT 0"),
        ("sus_done",   "INTEGER NOT NULL DEFAULT 0"),
        ("version",    "TEXT NOT NULL DEFAULT 'O'"),
        ("sus_score",  "REAL"),
        # active_version/current_round: the condition/round currently in
        # effect, separate from the permanent `version` -- lets the dev
        # preview toggle switch conditions mid-session for logging purposes.
        ("active_version",  "TEXT NOT NULL DEFAULT 'O'"),
        ("current_round",   "INTEGER NOT NULL DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass
    if "active_version" not in user_cols_before:
        conn.execute("UPDATE users SET active_version = version")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS demographics (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             INTEGER NOT NULL,
            age_group           TEXT,
            degree_job          TEXT,
            netflix_experience  INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id)
        )
    """)

    rating_cols = {r[1] for r in conn.execute("PRAGMA table_info(ratings)").fetchall()}
    if "version" in rating_cols:
        conn.execute("ALTER TABLE ratings RENAME TO _ratings_old")
        conn.execute("""
            CREATE TABLE ratings (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id  INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating   REAL    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, movie_id)
            )
        """)
        conn.execute(
            "INSERT OR IGNORE INTO ratings (user_id, movie_id, rating) "
            "SELECT user_id, movie_id, rating FROM _ratings_old"
        )
        conn.execute("DROP TABLE _ratings_old")
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id  INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating   REAL    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, movie_id)
            )
        """)

    sus_cols = {r[1] for r in conn.execute("PRAGMA table_info(sus_responses)").fetchall()}
    if "version" in sus_cols:
        conn.execute("ALTER TABLE sus_responses RENAME TO _sus_old")
        conn.execute("""
            CREATE TABLE sus_responses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                question_idx INTEGER NOT NULL,
                response     INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, question_idx)
            )
        """)
        conn.execute(
            "INSERT OR IGNORE INTO sus_responses (user_id, question_idx, response) "
            "SELECT user_id, question_idx, response FROM _sus_old"
        )
        conn.execute("DROP TABLE _sus_old")
    else:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sus_responses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                question_idx INTEGER NOT NULL,
                response     INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                UNIQUE(user_id, question_idx)
            )
        """)

    # Deltas, not absolute values -- stays meaningful as the AI-inferred
    # profile itself shifts with new ratings. No row = no override for that genre.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile_overrides (
            user_id INTEGER NOT NULL,
            genre   TEXT    NOT NULL,
            delta   REAL    NOT NULL,
            PRIMARY KEY (user_id, genre),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile_snapshots (
            user_id      INTEGER PRIMARY KEY,
            profile_json TEXT    NOT NULL,
            updated_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Append-only event log (unlike profile_overrides/profile_snapshots
    # above, which hold current state only). Each row is tagged with the
    # version/round active on the account when the action happened.

    conn.execute("""
        CREATE TABLE IF NOT EXISTS shown_movies (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL,
            version  TEXT    NOT NULL,
            round    INTEGER NOT NULL,
            movie_id INTEGER NOT NULL,
            shown_at TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS rating_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            version    TEXT    NOT NULL,
            round      INTEGER NOT NULL,
            movie_id   INTEGER NOT NULL,
            rating     REAL    NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile_edit_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            version    TEXT    NOT NULL,
            round      INTEGER NOT NULL,
            genre      TEXT    NOT NULL,
            delta      REAL    NOT NULL,
            level      TEXT    NOT NULL,
            source     TEXT    NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            version    TEXT    NOT NULL,
            round      INTEGER NOT NULL,
            trigger_type TEXT  NOT NULL,
            list_type  TEXT    NOT NULL,
            rank       INTEGER NOT NULL,
            movie_id   INTEGER NOT NULL,
            title      TEXT    NOT NULL,
            score      REAL    NOT NULL,
            created_at TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS condition_switches (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            version    TEXT    NOT NULL,
            switched_at TEXT   NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()
