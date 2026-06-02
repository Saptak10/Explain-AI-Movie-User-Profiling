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
            hashed_password  TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            movie_id  INTEGER NOT NULL,
            rating    REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, movie_id)
        )
    """)
    conn.commit()
    conn.close()
