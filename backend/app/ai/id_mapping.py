"""
id_mapping.py
-------------
ID Translation Layer for the ml-latest (33M rating) migration.

Responsibilities:
  1. Stream movies.csv exactly once to build a bidirectional mapping between
     arbitrary, discontinuous MovieLens movieIds (these jump past 200,000 in
     ml-latest) and dense, contiguous vector indices [0, N-1].
  2. Dynamically discover the genre vocabulary by splitting the pipe-separated
     genre strings at startup -- we do NOT hardcode a fixed 18/20-genre list,
     because ml-latest's actual genre set must be read from the file to be
     correct (and to satisfy the "(no genres listed)" edge case below).
  3. Build the prior-knowledge genre mask (the "pure genre matrix") used to
     initialize and clip the autoencoder's bottleneck-facing weights.

Movies labeled '(no genres listed)' are mapped to an explicit, namable token
that is EXCLUDED from the genre vocabulary itself, and they receive an
all-zero row in the prior-knowledge mask. This prevents semantic drift: such
a movie does not get spuriously associated with any real genre, but it still
takes a valid dense index and matrix row.

This module never loads ratings.csv. It is intentionally cheap: parsing
movies.csv (whether ~9k rows in ml-latest-small or ~87k rows in ml-latest) is
trivial memory-wise, and is done with the csv module rather than pandas so
there is exactly one well-defined place where 'movies.csv' is read.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

NO_GENRES_TOKEN = "(no genres listed)"


@dataclass
class IdMapping:
    """
    Holds the fully-built bidirectional movie ID <-> dense index mapping,
    the discovered genre vocabulary, and the prior-knowledge genre mask.

    Attributes:
        movie_id_to_idx: real MovieLens movieId -> dense index [0, N-1]
        idx_to_movie_id: dense index -> real MovieLens movieId
        idx_to_title:    dense index -> movie title (for XAI / API responses)
        genres:          sorted, deduplicated list of real genre tokens
                         (NO_GENRES_TOKEN is intentionally excluded from this
                         list -- it is not a genre, it is the absence of one)
        genre_to_idx:    genre token -> column index in the genre mask
        num_movies:      N, total number of distinct movies (dense index count)
        num_genres:      number of real genre columns
        genre_mask:      float32 list-of-lists, shape (num_movies, num_genres).
                         Row i is all zeros if movie i had no listed genres.
    """
    movie_id_to_idx: dict
    idx_to_movie_id: dict
    idx_to_title: dict
    genres: list
    genre_to_idx: dict
    num_movies: int
    num_genres: int
    genre_mask: list  # kept as plain nested lists here; converted to a
                       # numpy/torch tensor only at the model-construction
                       # boundary, so this module has zero torch dependency.

    def movie_id_to_dense(self, movie_id: int) -> int | None:
        """Translate a real movieId to its dense index, or None if unknown."""
        return self.movie_id_to_idx.get(movie_id)

    def dense_to_movie_id(self, dense_idx: int) -> int | None:
        """Translate a dense index back to its real movieId, or None if out of range."""
        return self.idx_to_movie_id.get(dense_idx)

    def title_for_dense(self, dense_idx: int) -> str:
        """Best-effort title lookup for a dense index, with a safe fallback."""
        return self.idx_to_title.get(dense_idx, f"Unknown Movie (idx={dense_idx})")


def _split_genres(raw_genre_field: str) -> list:
    """
    Splits a single movies.csv 'genres' cell on '|'.

    Returns an empty list for the no-genres sentinel or for any malformed/
    empty cell, so callers have one place to check "did this movie have any
    real genre" rather than special-casing the sentinel string everywhere.
    """
    raw_genre_field = raw_genre_field.strip()
    if raw_genre_field == "" or raw_genre_field == NO_GENRES_TOKEN:
        return []
    return [g for g in raw_genre_field.split("|") if g != ""]


def build_id_mapping(movies_csv_path: str) -> IdMapping:
    """
    Streams movies.csv exactly once (constant memory beyond the mapping
    dictionaries themselves, which scale with the number of movies, not
    ratings) and builds the complete IdMapping.

    Two passes over the *parsed rows* are made, but both happen from a single
    list built during the single file read -- we deliberately avoid reading
    the file from disk twice:
        Pass A (while reading): collect movieId, title, and raw genre list
                                 per row; accumulate the genre vocabulary.
        Pass B (in-memory):     now that num_genres / genre_to_idx is final,
                                 build the dense genre_mask rows.

    This two-pass-on-cached-rows approach is required because the genre
    vocabulary (and therefore the matrix's column count and column order)
    is only fully known after the entire file has been seen -- a genre
    appearing for the first time on the very last row still needs a column.

    Args:
        movies_csv_path: path to movies.csv (columns: movieId, title, genres)

    Returns:
        A fully populated IdMapping.

    Raises:
        FileNotFoundError: if movies_csv_path does not exist.
        ValueError: if movies.csv has no header or no data rows.
    """
    if not os.path.isfile(movies_csv_path):
        raise FileNotFoundError(f"movies.csv not found at: {movies_csv_path}")

    movie_id_to_idx: dict = {}
    idx_to_movie_id: dict = {}
    idx_to_title: dict = {}
    raw_genres_by_idx: dict = {}
    genre_vocab: set = set()

    with open(movies_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("movies.csv has no header row.")
        required_cols = {"movieId", "title", "genres"}
        missing = required_cols - set(reader.fieldnames)
        if missing:
            raise ValueError(f"movies.csv is missing required columns: {missing}")

        dense_idx = 0
        for row in reader:
            raw_movie_id = row["movieId"].strip()
            if raw_movie_id == "":
                continue  # defensively skip any malformed blank-id row
            movie_id = int(raw_movie_id)

            if movie_id in movie_id_to_idx:
                continue  # defensively guard against duplicate rows

            movie_id_to_idx[movie_id] = dense_idx
            idx_to_movie_id[dense_idx] = movie_id
            idx_to_title[dense_idx] = row["title"]

            genre_list = _split_genres(row["genres"])
            raw_genres_by_idx[dense_idx] = genre_list
            genre_vocab.update(genre_list)

            dense_idx += 1

    num_movies = dense_idx
    if num_movies == 0:
        raise ValueError("movies.csv contained a header but no data rows.")

    genres = sorted(genre_vocab)
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    num_genres = len(genres)

    genre_mask = [[0.0] * num_genres for _ in range(num_movies)]
    for idx, genre_list in raw_genres_by_idx.items():
        for g in genre_list:
            genre_mask[idx][genre_to_idx[g]] = 1.0
        # Movies with an empty genre_list (true '(no genres listed)' rows,
        # or any malformed empty genre cell) simply keep their all-zero row
        # from initialization above -- this *is* the zero-vector safeguard
        # against semantic drift required by the spec.

    return IdMapping(
        movie_id_to_idx=movie_id_to_idx,
        idx_to_movie_id=idx_to_movie_id,
        idx_to_title=idx_to_title,
        genres=genres,
        genre_to_idx=genre_to_idx,
        num_movies=num_movies,
        num_genres=num_genres,
        genre_mask=genre_mask,
    )
