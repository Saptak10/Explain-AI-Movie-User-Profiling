"""
xai.py
------
Human-Centered XAI utilities:

  1. hydrate_sparse_input: turns an API-facing sparse {movieId: rating}
     payload into a dense (1, num_movies) tensor via the IdMapping,
     immediately before the model's forward pass. This is the boundary
     where "sparse JSON in" becomes "dense tensor for torch" -- it is
     intentionally the *only* place a dense num_movies vector gets built
     from a request, and it is built fresh per-request rather than ever
     being cached globally.

  2. compute_local_feature_importance: a permutation-importance-style local
     explainer, deliberately restricted to the user's *active, non-zero*
     ratings rather than the full ~87k-dimensional movie space. Computing
     importance over every possible movie a user has never rated would be
     both meaningless (zeroing an already-zero input changes nothing) and
     a real-time API timeout risk at ml-latest scale; restricting to the
     non-zero support set keeps the computation proportional to a user's
     actual watch history (typically tens to low hundreds of movies), not
     to the size of the catalog.

  3. generate_soft_rationale: cross-references a user's non-zero inputs,
     the dominant bottleneck (genre) nodes, and the final top
     recommendation to produce a natural-language rationale string.
"""

from __future__ import annotations

import torch

from app.ai.id_mapping import IdMapping
from app.ai.model import DualModeHCAIAutoEncoder


def hydrate_sparse_input(
    sparse_payload: dict, id_mapping: IdMapping
) -> torch.Tensor:
    """
    Converts a sparse API payload, e.g. {"1": 4.5, "200123": 3.0}, into a
    dense (1, num_movies) float32 tensor suitable for
    DualModeHCAIAutoEncoder.forward_standard().

    movieId keys not present in id_mapping are skipped (with no error
    raised) rather than failing the whole request, since a client sending
    one stale or typo'd movieId should not block recommendations for every
    other rating they sent. Keys may be int or numeric-string (JSON object
    keys are always strings on the wire), both are accepted.

    Args:
        sparse_payload: {movieId (int or str): rating (float)}.
        id_mapping: the IdMapping used to translate movieId -> dense index.

    Returns:
        (1, num_movies) float32 tensor, zero everywhere except the
        translated rated positions.
    """
    vector = torch.zeros(1, id_mapping.num_movies, dtype=torch.float32)
    for raw_movie_id, rating in sparse_payload.items():
        movie_id = int(raw_movie_id)
        dense_idx = id_mapping.movie_id_to_dense(movie_id)
        if dense_idx is None:
            continue
        vector[0, dense_idx] = float(rating)
    return vector


def translate_topk_to_movie_ids(
    topk_indices: torch.Tensor, id_mapping: IdMapping
) -> list:
    """
    Translates a 1D tensor of dense top-X output indices back into real
    MovieLens movieIds, in the same order, for the JSON response.

    Args:
        topk_indices: 1D LongTensor of dense indices (e.g. from
                      torch.topk(...).indices).
        id_mapping: the IdMapping used to translate dense index -> movieId.

    Returns:
        List of real movieId ints, same length and order as topk_indices.
    """
    return [id_mapping.dense_to_movie_id(int(i.item())) for i in topk_indices]


def compute_local_feature_importance(
    model: DualModeHCAIAutoEncoder,
    sparse_input_vector: torch.Tensor,
    target_movie_idx: int,
) -> dict:
    """
    Optimized local explainability via leave-one-out permutation importance,
    restricted strictly to the user's non-zero (actively rated) movie
    positions -- never the full ~87k-dimensional input space.

    Methodology: for each non-zero position i in the user's rating vector,
    zero out only that one position, re-run forward_standard, and measure
    how much the predicted score for `target_movie_idx` changed. The
    magnitude of that change is the local importance of rating i to this
    particular recommendation.

    Cost: O(k) forward passes where k = number of the user's non-zero
    ratings (typically tens to a few hundred), not O(num_movies). This is
    what keeps the computation real-time-safe at ml-latest scale.

    Args:
        model: trained DualModeHCAIAutoEncoder, in eval() mode by caller
               convention (this function does not toggle train/eval modes
               itself, since callers may have specific dropout-behavior
               needs around batched calls).
        sparse_input_vector: (1, num_movies) float32, the user's actual
                              sparse rating vector (zeros for unrated
                              movies).
        target_movie_idx: dense index of the movie whose predicted score's
                           sensitivity to each rating is being explained.

    Returns:
        Dict {movie_dense_idx (int): importance (float)}, restricted to
        exactly the non-zero positions of sparse_input_vector, sorted by
        descending absolute importance.
    """
    with torch.no_grad():
        baseline_predictions, _ = model.forward_standard(sparse_input_vector)
        baseline_score = baseline_predictions[0, target_movie_idx].item()

        nonzero_positions = torch.nonzero(
            sparse_input_vector[0], as_tuple=False
        ).flatten().tolist()

        importances: dict = {}
        for pos in nonzero_positions:
            perturbed = sparse_input_vector.clone()
            perturbed[0, pos] = 0.0
            perturbed_predictions, _ = model.forward_standard(perturbed)
            perturbed_score = perturbed_predictions[0, target_movie_idx].item()
            importances[pos] = baseline_score - perturbed_score

    return dict(
        sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )


def generate_soft_rationale(
    sparse_input_vector: torch.Tensor,
    latent_profile: torch.Tensor,
    target_movie_idx: int,
    id_mapping: IdMapping,
    genre_mask_row: list,
    genre_dominance_threshold: float = 0.6,
    top_contributing_ratings: int = 1,
) -> str:
    """
    Soft Rationale Logic: cross-references the user's non-zero inputs, the
    dominant bottleneck (genre) nodes, and the target recommendation's own
    genre membership to produce a natural-language rationale string such
    as: "Recommended because your Action profile is high, heavily
    influenced by your 5-star rating of Iron Man."

    Args:
        sparse_input_vector: (1, num_movies) the user's sparse ratings.
        latent_profile: (num_genres,) or (1, num_genres) sigmoid-activated
                         bottleneck activations for this user.
        target_movie_idx: dense index of the recommended movie being
                           explained.
        id_mapping: used to resolve movie titles and the recommended
                    movie's own genre membership.
        genre_mask_row: the recommended movie's own genre row (length
                        num_genres, 1.0/0.0 per genre) -- i.e.
                        id_mapping.genre_mask[target_movie_idx].
        genre_dominance_threshold: minimum latent activation for a genre
                                    to be called "high" in the rationale.
        top_contributing_ratings: how many of the user's highest-rated,
                                   genre-overlapping movies to cite by name.

    Returns:
        A natural-language rationale string. Falls back to a generic
        collaborative-filtering rationale if no genre clears the dominance
        threshold and overlaps the target movie's own genres.
    """
    flat_profile = latent_profile.detach().reshape(-1)
    target_title = id_mapping.title_for_dense(target_movie_idx)

    dominant_genre_indices = [
        i
        for i in range(len(genre_mask_row))
        if genre_mask_row[i] > 0.0 and flat_profile[i].item() > genre_dominance_threshold
    ]

    if not dominant_genre_indices:
        return (
            f'"{target_title}" is recommended based on general collaborative '
            f"filtering patterns across similar users."
        )

    dominant_genre_names = [id_mapping.genres[i] for i in dominant_genre_indices]

    rated_positions = torch.nonzero(
        sparse_input_vector[0], as_tuple=False
    ).flatten().tolist()
    overlapping_rated = [
        pos
        for pos in rated_positions
        if any(id_mapping.genre_mask[pos][g] > 0.0 for g in dominant_genre_indices)
    ]
    overlapping_rated.sort(
        key=lambda pos: sparse_input_vector[0, pos].item(), reverse=True
    )
    top_cited = overlapping_rated[:top_contributing_ratings]

    genre_clause = ", ".join(dominant_genre_names)
    if not top_cited:
        return (
            f'Recommended because your {genre_clause} profile is high.'
        )

    citation_clauses = []
    for pos in top_cited:
        rating_value = sparse_input_vector[0, pos].item()
        movie_title = id_mapping.title_for_dense(pos)
        citation_clauses.append(f"your {rating_value:.1f}-star rating of {movie_title}")

    citation_text = " and ".join(citation_clauses)
    return (
        f"Recommended because your {genre_clause} profile is high, "
        f"heavily influenced by {citation_text}."
    )
