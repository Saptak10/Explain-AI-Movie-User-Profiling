"""
db.py
-----
deployment branch: Postgres (asyncpg) instead of SQLite. Render's free web
service has no persistent disk -- a local SQLite file gets wiped on every
cold-start restart (the free tier spins the instance down after ~15 min of
no traffic), which meant every account, rating, and survey response
vanished the moment a tester came back after a short gap. Render's free
Postgres plan is a separate managed service with its own persistent
storage, unaffected by the web service's own restarts.

Public API (execute/fetchone/fetchall/init_db) is unchanged from the
SQLite version, so route/service call sites didn't need to change except
for the handful of genuinely SQLite-specific SQL constructs (INSERT OR
REPLACE, datetime('now')) -- see ai_routes.py/sus_routes.py/
auth_service.py. '?' placeholders are translated to asyncpg's positional
$1, $2, ... automatically, so existing call sites keep working unmodified.
"""

import re

import asyncpg

_POOL: asyncpg.Pool | None = None
_DB_URL: str = ""


def _to_positional(sql: str) -> str:
    """Translates sqlite-style '?' placeholders to asyncpg's $1, $2, ... style."""
    counter = iter(range(1, sql.count("?") + 1))
    return re.sub(r"\?", lambda _: f"${next(counter)}", sql)


async def _get_pool() -> asyncpg.Pool:
    global _POOL
    if _POOL is None:
        # Render's managed Postgres requires SSL/TLS on every connection,
        # internal or external.
        _POOL = await asyncpg.create_pool(_DB_URL, min_size=1, max_size=5, ssl="require")
    return _POOL


async def execute(sql: str, params: tuple = ()) -> int | None:
    """
    Returns the first column of the first returned row for statements with
    a RETURNING clause (mirrors sqlite3's cursor.lastrowid for the one call
    site that needs the new row's id -- see auth_service.register), or None
    for statements with no result set.
    """
    pool = await _get_pool()
    sql = _to_positional(sql)
    async with pool.acquire() as conn:
        if "returning" in sql.lower():
            row = await conn.fetchrow(sql, *params)
            return row[0] if row else None
        await conn.execute(sql, *params)
        return None


async def fetchone(sql: str, params: tuple = ()) -> dict | None:
    pool = await _get_pool()
    sql = _to_positional(sql)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *params)
        return dict(row) if row else None


async def fetchall(sql: str, params: tuple = ()) -> list[dict]:
    pool = await _get_pool()
    sql = _to_positional(sql)
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
        return [dict(r) for r in rows]


async def init_db(database_url: str) -> None:
    global _DB_URL
    _DB_URL = database_url
    pool = await _get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id               SERIAL PRIMARY KEY,
                username         TEXT UNIQUE NOT NULL,
                hashed_password  TEXT NOT NULL,
                has_edited       INTEGER NOT NULL DEFAULT 0,
                sus_done         INTEGER NOT NULL DEFAULT 0,
                version          TEXT    NOT NULL DEFAULT 'O',
                edit_order       TEXT,
                sus_score        REAL
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS demographics (
                id                  SERIAL PRIMARY KEY,
                user_id             INTEGER NOT NULL REFERENCES users(id),
                age_group           TEXT,
                degree_job          TEXT,
                netflix_experience  INTEGER,
                UNIQUE(user_id)
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                id       SERIAL PRIMARY KEY,
                user_id  INTEGER NOT NULL REFERENCES users(id),
                movie_id INTEGER NOT NULL,
                rating   REAL    NOT NULL,
                round    INTEGER NOT NULL DEFAULT 0,
                UNIQUE(user_id, movie_id)
            )
        """)

        # Profile edits — log of every weight change a user makes, so it
        # can be linked back to the user, the round, and whether it was a
        # per-movie or whole-profile edit.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS profile_edits (
                id         SERIAL PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id),
                round      INTEGER NOT NULL,
                edit_type  TEXT    NOT NULL,
                genre      TEXT    NOT NULL,
                level      TEXT    NOT NULL,
                movie_id   INTEGER,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Recommendation sessions — what movies were shown, when, and in
        # what order so the research team can reconstruct each user's
        # recommendation experience per round and per edit type.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_sessions (
                id         SERIAL PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id),
                round      INTEGER NOT NULL,
                rec_type   TEXT    NOT NULL,
                movie_id   INTEGER NOT NULL,
                position   INTEGER NOT NULL,
                score      REAL    NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sus_responses (
                id           SERIAL PRIMARY KEY,
                user_id      INTEGER NOT NULL REFERENCES users(id),
                question_idx INTEGER NOT NULL,
                response     INTEGER NOT NULL,
                UNIQUE(user_id, question_idx)
            )
        """)

        # Profile overrides — persisted genre-preference deltas from the
        # Edit Profile UI, so a user's edits keep affecting /api/profile
        # and /api/recommend on every future visit instead of only the one
        # request they were made in. Stored as deltas (not absolute
        # values) relative to whatever the AI-inferred profile is at read
        # time. One row per (user, genre); absence of a row means "no
        # override for this genre."
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS profile_overrides (
                user_id INTEGER NOT NULL REFERENCES users(id),
                genre   TEXT    NOT NULL,
                delta   REAL    NOT NULL,
                PRIMARY KEY (user_id, genre)
            )
        """)

        # Profile snapshots — the latest AI-inferred (+ override-adjusted)
        # taste profile per user, saved as JSON whenever GET /api/profile
        # is served, so researchers can query a user's profile directly
        # from the database without re-deriving it through the API/model.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS profile_snapshots (
                user_id      INTEGER PRIMARY KEY REFERENCES users(id),
                profile_json TEXT    NOT NULL,
                updated_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
