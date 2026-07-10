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
    user_cols_before = {r[1] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
    for col, defn in [
        ("has_edited", "INTEGER NOT NULL DEFAULT 0"),
        ("sus_done",   "INTEGER NOT NULL DEFAULT 0"),
        ("version",    "TEXT NOT NULL DEFAULT 'O'"),
        ("sus_score",  "REAL"),
        # active_version/current_round track which A/B condition is *currently*
        # driving this account's UI and how many rate→recommend cycles it has
        # been through under that condition, so study-data logging (below) can
        # be attributed correctly even when an admin manually flips the
        # condition mid-session via the dev preview toggle instead of the
        # account's permanent `version` (assigned once at registration).
        ("active_version",  "TEXT NOT NULL DEFAULT 'O'"),
        ("current_round",   "INTEGER NOT NULL DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass
    # active_version's constant ALTER default can't copy the per-row `version`
    # value, so backfill it once, right after the column is first added, for
    # every account that already existed at that point.
    if "active_version" not in user_cols_before:
        conn.execute("UPDATE users SET active_version = version")

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

    # ── Profile overrides — persisted genre-preference deltas from the
    # Edit Profile UI, so a user's edits keep affecting /api/profile and
    # /api/recommend on every future visit instead of only the one request
    # they were made in. Stored as deltas (not absolute values) relative to
    # whatever the AI-inferred profile is at read time, matching the UI's
    # boost/suppress language and staying meaningful as the AI profile
    # itself shifts with new ratings. One row per (user, genre); absence of
    # a row means "no override for this genre."
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile_overrides (
            user_id INTEGER NOT NULL,
            genre   TEXT    NOT NULL,
            delta   REAL    NOT NULL,
            PRIMARY KEY (user_id, genre),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # ── Profile snapshots — the latest AI-inferred (+ override-adjusted)
    # taste profile per user, saved as JSON whenever GET /api/profile is
    # served, so researchers can query a user's profile directly from the
    # database without re-deriving it through the API/model.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profile_snapshots (
            user_id      INTEGER PRIMARY KEY,
            profile_json TEXT    NOT NULL,
            updated_at   TEXT    NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # ── Study event log — append-only tables recording every study-relevant
    # action as it happens, so it can be reconstructed later for the paper's
    # evaluation (unlike profile_overrides/profile_snapshots above, which only
    # ever hold the *current* state and overwrite on every edit). Each row is
    # tagged with the `version` (A/B condition: 'O' transparent / 'N'
    # standard) and `round` (0 = initial rating phase, 1 = after the first
    # recommend call, 2 = after the next, ...) active on the account at the
    # moment the action happened — see users.active_version/current_round.

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

    # One row per genre touched per edit action (not per current value --
    # see profile_overrides for that). `source` distinguishes an edit made
    # from a specific recommendation card ('movie_card') from one made on
    # the main Profile page ('profile_page').
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

    # One row per movie per recommendation list per recommend call.
    # `trigger` is 'initial' for a plain /recommend call or 'edited' for
    # /recommend/edited-profile, so a coworker can tell an unedited refresh
    # apart from a list generated right after an override was applied.
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

    # Audit trail of manual A/B condition switches (dev preview toggle),
    # so it's possible to see exactly when a within-subjects session moved
    # from one condition to the other.
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
