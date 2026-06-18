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

    # ── Users ─────────────────────────────────────────────────────────────────
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
    for col, defn in [
        ("has_edited", "INTEGER NOT NULL DEFAULT 0"),
        ("sus_done",   "INTEGER NOT NULL DEFAULT 0"),
        ("version",    "TEXT NOT NULL DEFAULT 'O'"),
    ]:
        try:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass

    # ── Demographics ──────────────────────────────────────────────────────────
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

    # ── Ratings — no version column ────────────────────────────────────────
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

    # ── SUS responses — no version column ─────────────────────────────────
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

    conn.commit()
    conn.close()
