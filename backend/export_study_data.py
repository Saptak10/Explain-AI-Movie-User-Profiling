"""
export_study_data.py
---------------------
Standalone export tool that pulls every study-relevant record out of the
app database and writes it to a set of coworker-ready CSV files (plus a
codebook explaining every column), zipped into one file for handoff.

Run from the backend/ directory:
    python export_study_data.py
    python export_study_data.py --output-dir export --db app/database/app.db

Produces, under --output-dir (default "export/"):
    users.csv               one row per registered account
    demographics.csv        age group / job or degree / Netflix experience
    ratings.csv             each user's current rating per movie (always complete)
    profile_overrides.csv   each user's current genre boost/suppress deltas (always complete)
    condition_switches.csv  every manual A/B condition switch (dev toggle)
    shown_movies.csv        every movie shown in the rate-page pool
    rating_events.csv       every rating submitted, with round + condition
    profile_edits.csv       every genre boost/suppress edit, with round,
                             condition, direction/magnitude, and which UI
                             surface (movie card vs. profile page) it came from
    recommendations.csv     every movie shown in a recommendation list, with
                             rank, score, list type, round, condition
    sus_responses.csv       every individual SUS question response
    user_summary.csv        one row per user aggregating the above into the
                             counts/breakdowns most likely needed directly
                             (total edits, edits by source, SUS score, ...)
    CODEBOOK.md             plain-language description of every file/column
    study_export_<ts>.zip   all of the above, zipped together

Gracefully degrades against a database from before the event-logging
tables existed -- those come back empty rather than erroring.

Read-only: opens the SQLite file and runs a plain SELECT per table.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from app.config.config import settings  # noqa: E402


def _load_movie_titles(movies_csv_path: str) -> dict[int, str]:
    titles = {}
    with open(movies_csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                titles[int(row["movieId"])] = row["title"]
            except (KeyError, ValueError):
                continue
    return titles


_missing_tables: list[str] = []


def _fetch_all(conn: sqlite3.Connection, sql: str) -> list[dict]:
    """Returns [] instead of raising if the table doesn't exist yet."""
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql).fetchall()]
    except sqlite3.OperationalError as e:
        table = sql.split("FROM", 1)[1].strip().split()[0]
        _missing_tables.append(table)
        return []


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


SUS_QUESTIONS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]


def export(db_path: str, movies_csv_path: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    titles = _load_movie_titles(movies_csv_path)
    conn = sqlite3.connect(db_path)

    users = _fetch_all(conn, "SELECT * FROM users ORDER BY id")
    username_by_id = {u["id"]: u["username"] for u in users}

    demographics = _fetch_all(conn, "SELECT * FROM demographics ORDER BY user_id")
    ratings = _fetch_all(conn, "SELECT * FROM ratings ORDER BY user_id, movie_id")
    overrides = _fetch_all(conn, "SELECT * FROM profile_overrides ORDER BY user_id, genre")
    condition_switches = _fetch_all(conn, "SELECT * FROM condition_switches ORDER BY user_id, id")
    shown_movies = _fetch_all(conn, "SELECT * FROM shown_movies ORDER BY user_id, id")
    rating_events = _fetch_all(conn, "SELECT * FROM rating_events ORDER BY user_id, id")
    profile_edits = _fetch_all(conn, "SELECT * FROM profile_edit_events ORDER BY user_id, id")
    recommendations = _fetch_all(conn, "SELECT * FROM recommendation_events ORDER BY user_id, id")
    sus_responses = _fetch_all(conn, "SELECT * FROM sus_responses ORDER BY user_id, question_idx")
    conn.close()

    for row in shown_movies:
        row["username"] = username_by_id.get(row["user_id"], "")
        row["title"] = titles.get(row["movie_id"], "")
    for row in rating_events:
        row["username"] = username_by_id.get(row["user_id"], "")
        row["title"] = titles.get(row["movie_id"], "")
    for row in profile_edits:
        row["username"] = username_by_id.get(row["user_id"], "")
    for row in recommendations:
        row["username"] = username_by_id.get(row["user_id"], "")
    for row in condition_switches:
        row["username"] = username_by_id.get(row["user_id"], "")
    for row in demographics:
        row["username"] = username_by_id.get(row["user_id"], "")
    for row in ratings:
        row["username"] = username_by_id.get(row["user_id"], "")
        row["title"] = titles.get(row["movie_id"], "")
    for row in overrides:
        row["username"] = username_by_id.get(row["user_id"], "")
    for row in sus_responses:
        row["username"] = username_by_id.get(row["user_id"], "")
        row["question_text"] = SUS_QUESTIONS[row["question_idx"]] if 0 <= row["question_idx"] < len(SUS_QUESTIONS) else ""

    _write_csv(output_dir / "users.csv", users,
               ["id", "username", "version", "active_version", "current_round",
                "has_edited", "sus_done", "sus_score"])
    _write_csv(output_dir / "demographics.csv", demographics,
               ["user_id", "username", "age_group", "degree_job", "netflix_experience"])
    _write_csv(output_dir / "ratings.csv", ratings,
               ["user_id", "username", "movie_id", "title", "rating"])
    _write_csv(output_dir / "profile_overrides.csv", overrides,
               ["user_id", "username", "genre", "delta"])
    _write_csv(output_dir / "condition_switches.csv", condition_switches,
               ["id", "user_id", "username", "version", "switched_at"])
    _write_csv(output_dir / "shown_movies.csv", shown_movies,
               ["id", "user_id", "username", "version", "round", "movie_id", "title", "shown_at"])
    _write_csv(output_dir / "rating_events.csv", rating_events,
               ["id", "user_id", "username", "version", "round", "movie_id", "title", "rating", "created_at"])
    _write_csv(output_dir / "profile_edits.csv", profile_edits,
               ["id", "user_id", "username", "version", "round", "genre", "delta", "level", "source", "created_at"])
    _write_csv(output_dir / "recommendations.csv", recommendations,
               ["id", "user_id", "username", "version", "round", "trigger_type", "list_type",
                "rank", "movie_id", "title", "score", "created_at"])
    _write_csv(output_dir / "sus_responses.csv", sus_responses,
               ["user_id", "username", "question_idx", "question_text", "response"])

    _write_user_summary(output_dir, users, profile_edits, ratings, shown_movies, recommendations, overrides)
    _write_codebook(output_dir)

    zip_path = output_dir / f"study_export_{datetime.now():%Y%m%d_%H%M%S}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for csv_file in output_dir.glob("*.csv"):
            zf.write(csv_file, csv_file.name)
        zf.write(output_dir / "CODEBOOK.md", "CODEBOOK.md")

    return zip_path


def _write_user_summary(output_dir, users, profile_edits, ratings, shown_movies, recommendations, overrides):
    edits_by_user = defaultdict(lambda: {"total": 0, "movie_card": 0, "profile_page": 0, "genres": set()})
    for e in profile_edits:
        s = edits_by_user[e["user_id"]]
        s["total"] += 1
        s[e["source"]] = s.get(e["source"], 0) + 1
        s["genres"].add(e["genre"])

    active_overrides_by_user = defaultdict(int)
    for o in overrides:
        active_overrides_by_user[o["user_id"]] += 1

    ratings_by_user = defaultdict(int)
    for r in ratings:
        ratings_by_user[r["user_id"]] += 1

    shown_by_user = defaultdict(int)
    for m in shown_movies:
        shown_by_user[m["user_id"]] += 1

    rec_calls_by_user = defaultdict(set)
    for r in recommendations:
        rec_calls_by_user[r["user_id"]].add((r["version"], r["round"]))

    rows = []
    for u in users:
        uid = u["id"]
        edits = edits_by_user[uid]
        rows.append({
            "user_id": uid,
            "username": u["username"],
            "assigned_version": u.get("version", ""),
            "current_active_version": u.get("active_version", u.get("version", "")),
            "current_round": u.get("current_round", 0),
            "movies_shown_count": shown_by_user[uid],
            "ratings_submitted_count": ratings_by_user[uid],
            "profile_edits_total": edits["total"],
            "profile_edits_from_movie_card": edits.get("movie_card", 0),
            "profile_edits_from_profile_page": edits.get("profile_page", 0),
            "distinct_genres_edited": len(edits["genres"]),
            "genres_currently_overridden": active_overrides_by_user[uid],
            "recommendation_rounds_count": len(rec_calls_by_user[uid]),
            "has_edited_flag": u.get("has_edited", ""),
            "sus_done": u.get("sus_done", ""),
            "sus_score": u.get("sus_score", ""),
        })

    _write_csv(output_dir / "user_summary.csv", rows, list(rows[0].keys()) if rows else [
        "user_id", "username", "assigned_version", "current_active_version", "current_round",
        "movies_shown_count", "ratings_submitted_count", "profile_edits_total",
        "profile_edits_from_movie_card", "profile_edits_from_profile_page",
        "distinct_genres_edited", "genres_currently_overridden", "recommendation_rounds_count",
        "has_edited_flag", "sus_done", "sus_score",
    ])


def _write_codebook(output_dir: Path) -> None:
    text = """# Study data export -- codebook

Every table is one CSV file in this export. `round` and `version` appear
in most of them -- read this section first.

## `version` (A/B condition)
`O` = Transparent AI (explanations + genre editing shown).
`N` = Standard AI (recommendations only, no explanations or editing).
Assigned once per account at registration (`users.version`, permanent).
An admin can *preview* the other condition for the same account mid-session
via a dev-only toggle -- when that happens, `users.active_version` changes
and every event from that point on is tagged with the new active condition,
not the original permanent one. `condition_switches.csv` logs every time
this happened. Compare a user's `version` (users.csv) against the `version`
column on their individual event rows if you need to detect this.

## `round`
Counts rate/recommend cycles *within the currently active condition*,
resetting to 0 every time the condition is switched:
  - `0` = the initial rating phase, before any recommendation list has
    been generated yet (ratings submitted here get round = 0).
  - `1` = after the first recommendation list was generated. Any edits
    made in response to that list, and the ratings submitted afterward
    (if any), are logged as round = 1.
  - `2`, `3`, ... = after each subsequent recommend call (e.g. the list
    regenerated after applying an edit).
`round` on a `recommendations.csv` row is the round *that list itself
represents* (i.e. it already reflects the call that produced it).

## Files

### users.csv
One row per account. `version` = permanent assigned condition.
`active_version`/`current_round` = condition/round currently in effect
(only differs from `version` if a dev preview switch happened).
`sus_score` = final 0-100 System Usability Scale score (Brooke 1996
formula), null until all 10 questions are answered.

### demographics.csv
Self-reported age group, degree/job, and prior Netflix experience,
collected together with the SUS survey.

### condition_switches.csv
One row per manual A/B condition switch. Use this to split a single
account's other event rows into "before"/"after" a switch if needed.

### shown_movies.csv
One row per movie that appeared in the rate-page's random sample pool,
in whichever round/condition it was shown. This is the full candidate
pool the user *could* have rated, not just the ones they did -- compare
against rating_events.csv to see selection rate.

### ratings.csv
One row per (user, movie) -- each user's *current* rating for every
movie they've rated, straight from the table that actually drives their
recommendations. Always complete. Use this for "which movies did each
user rate" and "what rating did they give it"; use rating_events.csv
(below) only if you need the round/condition/timing of when a rating
was submitted.

### rating_events.csv
One row per rating action (a re-rate of the same movie creates a new
row here, unlike ratings.csv above, which only keeps each movie's
latest rating). **Only complete for ratings submitted after this
logging was added** -- any session that rated movies before then has
real ratings in ratings.csv but no corresponding rows here, so
user_summary.csv's ratings_submitted_count is computed from ratings.csv,
not this file.

### profile_overrides.csv
One row per (user, genre) currently overridden -- each genre's *latest*
saved delta, straight from the table that actually drives that user's
recommendations. Always complete, same relationship to profile_edits.csv
as ratings.csv has to rating_events.csv: this is the current state (one
row per genre, overwritten on every re-edit), profile_edits.csv is the
full history of edit actions (only complete post-logging). If a genre
isn't listed here for a user, it has no active override (AI-inferred
value used as-is). user_summary.csv's genres_currently_overridden count
is computed from this file.

### profile_edits.csv
One row per genre touched by one edit action (clicking a boost/suppress
button and then Apply/Get Recommendations). `delta` is the numeric
change on the model's [0,1] genre scale (-0.5 to +0.5); `level` is its
human label (`strong_decrease`, `slight_decrease`, `neutral`,
`slight_increase`, `strong_increase`). `source` is `movie_card` (edited
from a specific recommendation's "Edit Preferences" panel) or
`profile_page` (edited from the main Profile page). A user re-editing
the same genre twice produces two rows here, even though only the
latest value is kept in the live app's own state.

### recommendations.csv
One row per movie per recommendation list per recommend call --
i.e. a full impression log. `list_type` is `top_rated` (highest raw
predicted score, may include broadly popular titles) or `for_you`
(ranked by personalization lift over a genre-neutral baseline, favors
movies specifically matched to this user). `trigger_type` is `initial`
(a plain, un-edited recommend call) or `edited` (generated right after
applying a genre override). `rank` is 1-indexed position within that
list. `score` is the model's raw predicted rating (0-5 scale).

### sus_responses.csv
One row per individual SUS question response (1-5 Likert scale).
`question_idx` is 0-indexed into the standard 10-question SUS
instrument; odd/even index determines whether the item is
reverse-scored in the total (see users.sus_score for the final result).

### user_summary.csv
One row per user, aggregating the tables above into the counts most
likely needed directly for the paper's results section: total ratings,
total profile edits (and the movie-card vs. profile-page split),
distinct genres touched, and number of distinct recommendation rounds
generated, alongside the SUS score.
"""
    (output_dir / "CODEBOOK.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=settings.db_path, help="Path to the SQLite database file")
    parser.add_argument("--movies-csv", default=settings.movies_csv_path, help="Path to movies.csv")
    parser.add_argument("--output-dir", default="export", help="Directory to write CSVs + zip into")
    args = parser.parse_args()

    zip_path = export(args.db, args.movies_csv, Path(args.output_dir))
    print(f"Wrote study data export to: {zip_path.resolve()}")
    if _missing_tables:
        print(
            "\nNote: this database predates some study-logging tables, so the "
            "following came back empty (users.active_version/current_round also "
            "fall back to just `version`/0 if missing): " + ", ".join(sorted(set(_missing_tables)))
        )


if __name__ == "__main__":
    main()
