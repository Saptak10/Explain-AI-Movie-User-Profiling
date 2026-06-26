"""
data_pipeline.py
-----------------
Memory-safe streaming data pipeline for ratings.csv at ml-latest scale
(~33M rows, ~330k users). No global pandas.read_csv. No full rating matrix
is ever materialized in RAM.

Core assumption (stated explicitly, not hidden): every official MovieLens
ratings.csv export -- including ml-latest -- is sorted by userId ascending
(and by timestamp within a user). This is documented in the dataset's
README and is what makes single-pass user-by-user aggregation correct
without a pre-sort step. RowAggregator below does NOT silently trust this:
it actively detects an out-of-order userId and raises, rather than quietly
producing corrupted (split) user vectors.

Pipeline shape:
    ratings.csv (disk, streamed line-by-line)
        -> RatingsStreamReader      (decodes + translates each row)
        -> UserAggregator           (buffers one user's (movie_idx, rating)
                                      pairs at a time; flushes on user change)
        -> SparseUserVectorDataset  (PyTorch IterableDataset; turns each
                                      flushed user into a dense num_movies
                                      vector + mask, lazily, one user at a
                                      time)
        -> DataLoader w/ batch_size (standard PyTorch batching; only
                                      BATCH_SIZE dense vectors ever exist
                                      in RAM simultaneously)

Peak RAM for the dataset/pipeline itself is therefore O(batch_size *
num_movies), not O(num_users * num_movies) -- the latter would be the
~330k * 87k * 4 bytes ~= 115 GB dense matrix the old prototype effectively
built via pandas + a full R_matrix, which is exactly what cannot fit in
16GB and what this module replaces.
"""

from __future__ import annotations

import csv
import random
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from app.ai.id_mapping import IdMapping


class RatingsStreamReader:
    """
    Streams ratings.csv row by row, translating each real movieId into its
    dense index via the supplied IdMapping. Rows whose movieId is not in
    the mapping (e.g. a ratings.csv referencing a movie absent from
    movies.csv) are skipped rather than crashing the whole pipeline, since
    a single bad foreign key should not abort a 33M-row training run.

    This class holds only one open file handle and one CSV row in flight
    at a time -- it never buffers the file's contents.
    """

    def __init__(self, ratings_csv_path: str, id_mapping: IdMapping):
        self.ratings_csv_path = ratings_csv_path
        self.id_mapping = id_mapping

    def __iter__(self) -> Iterator[tuple]:
        """Yields (user_id: int, movie_dense_idx: int, rating: float) tuples."""
        with open(self.ratings_csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("ratings.csv has no header row.")
            required_cols = {"userId", "movieId", "rating"}
            missing = required_cols - set(reader.fieldnames)
            if missing:
                raise ValueError(f"ratings.csv is missing required columns: {missing}")

            for row in reader:
                raw_user_id = row["userId"].strip()
                raw_movie_id = row["movieId"].strip()
                raw_rating = row["rating"].strip()
                if not raw_user_id or not raw_movie_id or not raw_rating:
                    continue

                user_id = int(raw_user_id)
                movie_id = int(raw_movie_id)
                dense_idx = self.id_mapping.movie_id_to_dense(movie_id)
                if dense_idx is None:
                    continue  # foreign-key row with no matching movie; skip

                rating = float(raw_rating)
                yield user_id, dense_idx, rating


class UserAggregator:
    """
    Consumes the (user_id, movie_dense_idx, rating) stream and groups
    consecutive rows sharing the same user_id into a single per-user
    record: (user_id, [(movie_dense_idx, rating), ...]).

    Relies on ratings.csv being sorted by userId (true for all official
    MovieLens exports). Detects a userId reappearing after a different
    userId has already been flushed -- which would silently corrupt
    aggregation by splitting one user's ratings across two batches -- and
    raises immediately so the problem surfaces during development rather
    than producing quietly-wrong training data on a 33M-row file.
    """

    def __init__(self, row_stream: Iterator[tuple]):
        self.row_stream = row_stream

    def __iter__(self) -> Iterator[tuple]:
        current_user_id = None
        current_ratings: list = []
        seen_user_ids: set = set()

        for user_id, dense_idx, rating in self.row_stream:
            if current_user_id is None:
                current_user_id = user_id
                seen_user_ids.add(user_id)

            if user_id == current_user_id:
                current_ratings.append((dense_idx, rating))
                continue

            if user_id in seen_user_ids:
                raise ValueError(
                    f"ratings.csv is not sorted by userId: userId={user_id} "
                    f"reappeared after userId={current_user_id} had already "
                    f"been flushed. User-by-user streaming aggregation "
                    f"requires a userId-sorted file; re-sort ratings.csv "
                    f"(e.g. `sort -t, -k1,1n`) before streaming it."
                )

            # user boundary: flush the completed user, then start the next
            yield current_user_id, current_ratings
            current_user_id = user_id
            current_ratings = [(dense_idx, rating)]
            seen_user_ids.add(user_id)

        if current_user_id is not None and current_ratings:
            yield current_user_id, current_ratings


def hydrate_user_vector(
    ratings_for_user: list, num_movies: int
) -> tuple:
    """
    Converts one user's sparse [(movie_dense_idx, rating), ...] list into a
    dense (num_movies,) float32 tensor plus a matching boolean mask tensor
    (True where the user actually rated that movie).

    This is the *only* point in the streaming pipeline where a dense,
    num_movies-length vector is materialized, and only one such vector
    exists per user at a time inside this function -- it is immediately
    handed off and not retained.

    Args:
        ratings_for_user: list of (movie_dense_idx, rating) for one user.
        num_movies: total number of movies (dense vector length).

    Returns:
        (rating_vector, mask_vector): both shape (num_movies,), dtype
        float32 and bool respectively.
    """
    rating_vector = torch.zeros(num_movies, dtype=torch.float32)
    mask_vector = torch.zeros(num_movies, dtype=torch.bool)
    for movie_idx, rating in ratings_for_user:
        rating_vector[movie_idx] = rating
        mask_vector[movie_idx] = True
    return rating_vector, mask_vector


def apply_random_masking(
    mask_vector: torch.Tensor, mask_fraction: float, generator: random.Random | None = None
) -> tuple:
    """
    Self-supervised masking: given a user's "known ratings" mask, randomly
    hides `mask_fraction` of the True positions to create a held-out
    evaluation mask, leaving the rest visible as model input.

    Used to turn the autoencoder into a predictive network: the model only
    ever sees `visible_mask`-selected ratings as nonzero input, and is
    scored strictly against `hidden_mask`-selected (held-out) ratings while
    completely ignoring positions the user never rated at all (mask_vector
    False everywhere) -- those were never candidates for hiding in the
    first place since you cannot hide a rating that does not exist.

    Args:
        mask_vector: (num_movies,) bool tensor, True where user has a
                     real rating.
        mask_fraction: fraction in [0, 1) of the True positions to hide.
        generator: optional random.Random instance for reproducibility in
                   tests; a fresh module-level random is used otherwise.

    Returns:
        (visible_mask, hidden_mask): both (num_movies,) bool tensors,
        disjoint, with visible_mask | hidden_mask == mask_vector exactly.
    """
    if not (0.0 <= mask_fraction < 1.0):
        raise ValueError(f"mask_fraction must be in [0, 1), got {mask_fraction}")

    rng = generator if generator is not None else random.Random()

    rated_positions = torch.nonzero(mask_vector, as_tuple=False).flatten().tolist()
    num_to_hide = int(round(len(rated_positions) * mask_fraction))
    hidden_positions = set(rng.sample(rated_positions, num_to_hide)) if num_to_hide > 0 else set()

    hidden_mask = torch.zeros_like(mask_vector)
    visible_mask = mask_vector.clone()
    for pos in hidden_positions:
        hidden_mask[pos] = True
        visible_mask[pos] = False

    return visible_mask, hidden_mask


class SparseUserVectorDataset(IterableDataset):
    """
    PyTorch IterableDataset that streams ratings.csv end-to-end: read rows
    -> aggregate per user -> hydrate one dense vector per user -> apply
    self-supervised masking -> yield.

    Because this is an IterableDataset (not a Dataset with __len__ and
    __getitem__), PyTorch's DataLoader never tries to index into a
    preloaded in-memory array -- it just pulls from this iterator and
    batches via the default collate_fn, which is exactly the streaming
    behavior the 16GB constraint requires.

    Each yielded sample is a dict so DataLoader's default collation stacks
    same-named tensors across the batch automatically:
        {
          "input_vector": (num_movies,) float32  -- masked input (visible
                            ratings only, hidden ones zeroed out, unrated
                            items zero as always)
          "target_vector": (num_movies,) float32 -- ground-truth ratings
                            (used together with hidden_mask to compute loss
                            only on held-out, previously-known ratings)
          "hidden_mask": (num_movies,) bool      -- True only at positions
                            that were hidden this draw; loss is computed
                            strictly here per the masked self-supervised
                            objective.
          "user_id": int                          -- original MovieLens
                            userId, for traceability/debugging.
        }
    """

    def __init__(
        self,
        ratings_csv_path: str,
        id_mapping: IdMapping,
        mask_fraction: float = 0.2,
        seed: int | None = None,
    ):
        super().__init__()
        self.ratings_csv_path = ratings_csv_path
        self.id_mapping = id_mapping
        self.mask_fraction = mask_fraction
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed) if self.seed is not None else random.Random()

        reader = RatingsStreamReader(self.ratings_csv_path, self.id_mapping)
        aggregator = UserAggregator(iter(reader))

        for user_id, ratings_for_user in aggregator:
            target_vector, known_mask = hydrate_user_vector(
                ratings_for_user, self.id_mapping.num_movies
            )
            visible_mask, hidden_mask = apply_random_masking(
                known_mask, self.mask_fraction, generator=rng
            )

            input_vector = target_vector.clone()
            input_vector[hidden_mask] = 0.0  # the model must not see hidden ratings

            yield {
                "input_vector": input_vector,
                "target_vector": target_vector,
                "hidden_mask": hidden_mask,
                "user_id": user_id,
            }
