"""
ai_service.py
--------------
Singleton service bridging FastAPI routes to the real app/ai modules:
id_mapping.py, onnx_model.py, xai.py.

deployment branch: this variant never trains and never imports torch. It
only serves a checkpoint that was already converted to ONNX on `main` (see
export_onnx.py) via onnxruntime + numpy. `import torch` alone costs ~200MB
of resident memory before touching any data, which -- combined with the
checkpoint itself -- was the dominant cost keeping the full-PyTorch service
over a 512MB free-tier ceiling even after every other memory optimization
(meta-device loading, releasing the checkpoint dict early, etc.).

create_personalized_profile() is NOT available here: it does real
gradient-descent fine-tuning at request time, which is fundamentally
incompatible with an inference-only ONNX runtime (no backprop). It raises
NotImplementedError, which ai_routes.py turns into a 501 -- the frontend
already falls back to the standard (non-personalized) profile gracefully
when that call fails (see RatingsPage.jsx's handleSave), so this is a
clean, non-breaking degradation, not a dead end.
"""

import json
import random

import numpy as np

from app.ai.id_mapping import IdMapping
from app.ai.onnx_model import OnnxDualModeModel
from app.ai.xai import (
    hydrate_sparse_input,
    compute_local_feature_importance,
    compute_genre_feature_importance,
    generate_soft_rationale,
)


class AIService:
    def __init__(self):
        self.model: OnnxDualModeModel = None
        self.id_mapping: IdMapping = None
        self.popular_movies: list = []
        self.global_importance: dict = {}
        # Genre-neutral baseline prediction per movie -- see _init(). Used to
        # rank recommendations by personalization "lift" instead of raw
        # score, so movies with a strong genre-independent decoder bias
        # (documentaries were the clearest case: several ranked in the
        # global top-20 even against a fully neutral 50%-every-genre input)
        # don't dominate every user's recommendations regardless of fit.
        self.baseline_predictions: np.ndarray = None
        self._onnx_dir: str = ""

    def setup(self, onnx_dir: str, movies_csv_path: str = "", ratings_csv_path: str = "") -> None:
        """
        movies_csv_path/ratings_csv_path are accepted (and ignored) so
        main.py's startup call site doesn't need branch-specific logic --
        this deployment never trains, so it never reads either CSV.
        """
        self._onnx_dir = onnx_dir

    @property
    def genres(self) -> list:
        """Read-only access to the dynamically discovered genre vocabulary."""
        return self.id_mapping.genres if self.id_mapping is not None else []

    @property
    def num_movies(self) -> int:
        return self.id_mapping.num_movies if self.id_mapping is not None else 0

    @property
    def num_genres(self) -> int:
        return self.id_mapping.num_genres if self.id_mapping is not None else 0

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Loads id_mapping.json + the two ONNX graphs from self._onnx_dir
        (populated by main.py's startup download step, see
        settings.model_download_url).
        """
        with open(f"{self._onnx_dir}/id_mapping.json") as f:
            raw = json.load(f)

        # JSON object keys are always strings on the wire; movie_id_to_idx
        # and idx_to_movie_id are keyed by real MovieLens ids / dense
        # indices respectively, both ints everywhere else in this codebase.
        id_mapping = IdMapping(
            movie_id_to_idx={int(k): v for k, v in raw["movie_id_to_idx"].items()},
            idx_to_movie_id={int(k): v for k, v in raw["idx_to_movie_id"].items()},
            idx_to_title={int(k): v for k, v in raw["idx_to_title"].items()},
            genres=raw["genres"],
            genre_to_idx=raw["genre_to_idx"],
            num_movies=raw["num_movies"],
            num_genres=raw["num_genres"],
            genre_mask=raw["genre_mask"],
        )

        model = OnnxDualModeModel(
            standard_path=f"{self._onnx_dir}/model_standard.onnx",
            interactive_path=f"{self._onnx_dir}/model_interactive.onnx",
            num_movies=id_mapping.num_movies,
            num_genres=id_mapping.num_genres,
        )

        self._init(model, id_mapping)

    def _init(self, model: OnnxDualModeModel, id_mapping: IdMapping) -> None:
        self.model = model
        self.id_mapping = id_mapping

        # Popular-movies and global-importance summaries are computed from
        # the model's learned weights only -- no dense ratings matrix is
        # read or held here.
        with np.load(f"{self._onnx_dir}/importance_aux.npz") as aux:
            imp = aux["global_importance"]
            movie_salience = aux["movie_salience"]

        self.global_importance = {
            g: float(imp[i]) for i, g in enumerate(id_mapping.genres)
        }

        # Cached pool is much larger than any one page shown to a user
        # (get_popular_sample below draws a random subset per request) --
        # this is what lets the Rate page's "Refresh Suggestions" surface a
        # genuinely different batch instead of the same fixed 50 every
        # time, for users whose first batch didn't include enough movies
        # they've actually seen to rate.
        top_idx = np.argsort(-np.abs(movie_salience))[:500].tolist()
        self.popular_movies = [
            {
                "id": id_mapping.dense_to_movie_id(i),
                "title": id_mapping.title_for_dense(i),
            }
            for i in top_idx
        ]

        self.baseline_predictions = self._compute_baseline(model)
        print("AI service ready.")

    def _compute_baseline(self, model: OnnxDualModeModel) -> np.ndarray:
        """
        Runs the decoder on a fully genre-neutral profile (0.5 for every
        genre -- the "no distinguishing signal at all" input under this
        architecture's Bayesian-smoothed calibration, see model.py) to get
        each movie's baseline predicted score independent of any user's
        genre profile. This isolates the decoder's per-movie bias term
        (learned generic appeal) from genre-driven signal -- ranking by
        raw predicted score conflates the two, which is exactly why movies
        with a high bias but poor genre fit (documentaries were the
        clearest case) were being recommended regardless of a user's
        actual profile.
        """
        neutral = np.full((1, model.num_genres), 0.5, dtype=np.float32)
        baseline = model.forward_interactive(neutral)
        return baseline[0]

    def get_popular_sample(self, exclude_ids: set = None, count: int = 50) -> list:
        """
        Random sample of `count` movies from the cached popularity pool
        (see _init, ~500 movies), excluding any already-shown IDs so
        "Refresh Suggestions" on the Rate page surfaces a genuinely
        different batch instead of repeating the same movies -- useful for
        a user whose first batch didn't include enough movies they've
        actually seen to build a meaningful profile from. Falls back to
        allowing repeats if excluding leaves too few candidates (e.g. the
        user has refreshed enough times to exhaust the pool), rather than
        ever returning fewer than requested.
        """
        exclude_ids = exclude_ids or set()
        candidates = [m for m in self.popular_movies if m["id"] not in exclude_ids]
        if len(candidates) < count:
            candidates = self.popular_movies
        return random.sample(candidates, min(count, len(candidates)))

    def search_movies(self, query: str, limit: int = 20) -> list:
        """
        Case-insensitive substring search over every movie's title (not just
        the ~500-title popularity pool), so a user can find and rate a
        specific movie by name on the Rate page even if it wasn't surfaced
        in their random popular-sample batch. Titles starting with the query
        are ranked above titles that merely contain it (e.g. "Matrix" finds
        "Matrix, The (1999)" before "Animatrix, The (2003)").
        """
        query = query.strip().lower()
        if not query:
            return []

        starts, contains = [], []
        for idx, title in self.id_mapping.idx_to_title.items():
            title_lower = title.lower()
            if title_lower.startswith(query):
                starts.append((idx, title))
            elif query in title_lower:
                contains.append((idx, title))

        starts.sort(key=lambda pair: pair[1])
        contains.sort(key=lambda pair: pair[1])
        results = (starts + contains)[:limit]
        return [
            {"id": self.id_mapping.dense_to_movie_id(idx), "title": title}
            for idx, title in results
        ]

    # ── Inference ────────────────────────────────────────────────────────────

    def _merge_overrides_into_latent(
        self, latent_profile: np.ndarray, overrides: dict
    ) -> np.ndarray:
        """
        Applies persisted/just-submitted genre-preference deltas on top of
        the AI-inferred (1, num_genres) latent profile, clamped back into
        [0, 1]. Returns a new array -- never mutates latent_profile in
        place, since callers may still need the un-overridden version.
        """
        if not overrides:
            return latent_profile
        merged = latent_profile.copy()
        for genre_name, delta in overrides.items():
            genre_idx = self.id_mapping.genre_to_idx.get(genre_name)
            if genre_idx is not None:
                merged[0, genre_idx] = np.clip(
                    merged[0, genre_idx] + float(delta), 0.0, 1.0
                )
        return merged

    def get_profile(self, ratings: dict, overrides: dict = None) -> dict:
        """
        Standard-mode genre taste profile: hydrates the sparse ratings dict,
        runs the full encoder -> bottleneck pass, applies any persisted
        genre-preference overrides (see profile_overrides table /
        ai_routes.py), and returns a {genre_name: percentage} dict via
        extract_taste_profile -- so a user's past edits keep showing up as
        "their" profile on every future visit, not just the one request
        they were made in.
        """
        vec = hydrate_sparse_input(ratings, self.id_mapping)
        _, latent_profile = self.model.forward_standard(vec)
        latent_profile = self._merge_overrides_into_latent(latent_profile, overrides)
        return self.model.extract_taste_profile(latent_profile[0], self.id_mapping.genres)

    def explain_profile(self, ratings: dict, top_genres: int = None, overrides: dict = None) -> dict:
        """
        Human-Centered XAI for the Taste Profile page: the taste profile
        itself (same as get_profile, overrides included), plus, for every
        genre (or the top `top_genres` if given), the movies they rated
        that most drove that genre's score -- answering "why is my {genre}
        score what it is?" the same way explain_movie answers "why was this
        movie recommended?".

        The citations themselves are computed against the AI's genuine,
        un-overridden signal (compute_genre_feature_importance below runs
        against the raw encoder output) even when overrides shift which
        genres end up in the top N or what percentage is displayed --
        overrides are a user-driven adjustment layer, not something for the
        model to "explain" a rating-based cause for.

        Returns:
            {
              "profile": {genre_name: percentage, ...},
              "genre_explanations": {
                genre_name: [{"movie_id": int, "title": str, "importance": float}, ...],
                ...
              }
            }
        """
        vec = hydrate_sparse_input(ratings, self.id_mapping)
        _, latent_profile = self.model.forward_standard(vec)
        display_profile = self._merge_overrides_into_latent(latent_profile, overrides)
        profile = self.model.extract_taste_profile(display_profile[0], self.id_mapping.genres)

        genre_explanations = {}
        for genre_name in list(profile.keys())[:top_genres]:
            genre_idx = self.id_mapping.genre_to_idx.get(genre_name)
            if genre_idx is None:
                continue
            importances_by_idx = compute_genre_feature_importance(self.model, vec, genre_idx)
            genre_explanations[genre_name] = [
                {
                    "movie_id": self.id_mapping.dense_to_movie_id(idx),
                    "title": self.id_mapping.title_for_dense(idx),
                    "importance": round(float(value), 4),
                }
                for idx, value in list(importances_by_idx.items())[:5]
            ]

        return {"profile": profile, "genre_explanations": genre_explanations}

    # Quality floor for lift-ranking (see _top_n_from_scores) -- a candidate
    # must be predicted at least this well-liked before "lift" is allowed
    # to influence its rank at all. Pure lift with no floor has a real
    # failure mode: a genuinely bad movie with an equally-low genre-neutral
    # baseline can still show *positive* lift (it merely underperforms its
    # own low expectations by less than average), which would rank it
    # above genuinely good, broadly-loved movies whose lift is near zero
    # precisely because they're already correctly predicted well for
    # everyone. Confirmed by testing: unconditional lift-ranking surfaced
    # "Birdemic: Shock and Terror" and similar near-unwatchable movies
    # ahead of far better predictions.
    MIN_SCORE_FOR_LIFT_RANKING = 3.0

    def _top_n_from_scores(
        self,
        scores: np.ndarray,
        rated_movie_ids: set,
        top_n: int,
        baseline: np.ndarray = None,
    ) -> list:
        """
        Shared candidate-ranking tail used by every inference path that ends
        in "top_n scored movies, excluding ones the user already rated":
        get_recommendations calls this instead of duplicating the same
        argsort/exclude/translate loop.

        If `baseline` is given (see _compute_baseline), candidates that
        clear MIN_SCORE_FOR_LIFT_RANKING are *ranked* by personalization
        lift (scores - baseline) rather than raw score, so that among
        movies already predicted to be genuinely well-liked, ones matched
        to this user's *specific* profile outrank ones that are merely
        universally high-scoring regardless of fit (movies with a strong
        genre-independent decoder bias -- see _compute_baseline's
        docstring -- would otherwise dominate every user's top-N
        regardless of genre alignment). Candidates below the quality floor
        are ranked by raw score only, at the bottom, so lift can never
        promote a poorly-predicted movie above a well-predicted one. The
        *displayed* "score" is always the raw predicted rating either way,
        since that's the number actually meaningful to a user.
        """
        if baseline is None:
            rank_scores = scores
        else:
            passes_floor = scores >= self.MIN_SCORE_FOR_LIFT_RANKING
            lift = scores - baseline
            # Below-floor candidates get their raw score minus a large
            # constant, so they always sort after every above-floor
            # candidate (which are ranked amongst themselves by lift) while
            # still preserving relative order within the below-floor group.
            rank_scores = np.where(passes_floor, lift, scores - 100.0)
        candidate_order = np.argsort(-rank_scores)

        results = []
        for dense_idx in candidate_order.tolist():
            real_movie_id = self.id_mapping.dense_to_movie_id(dense_idx)
            if real_movie_id in rated_movie_ids:
                continue
            results.append({
                "movie_id": real_movie_id,
                "title": self.id_mapping.title_for_dense(dense_idx),
                "score": round(float(scores[dense_idx]), 3),
            })
            if len(results) >= top_n:
                break
        return results

    def _rank_both(
        self, scores: np.ndarray, rated_movie_ids: set, top_n: int, baseline: np.ndarray
    ) -> dict:
        """
        Produces both recommendation orderings from a single, already-computed
        predictions array -- no extra model forward pass needed, just two
        argsorts over the same scores:
          - "top_rated": ranked by raw predicted score, the movies this user
            is predicted to rate the highest, period (may include broadly
            popular titles that aren't specifically matched to this user).
          - "for_you": ranked by lift with a quality floor (see
            _top_n_from_scores/MIN_SCORE_FOR_LIFT_RANKING), movies matched
            to this user's specific profile rather than generic appeal.
        """
        return {
            "top_rated": self._top_n_from_scores(scores, rated_movie_ids, top_n, baseline=None),
            "for_you": self._top_n_from_scores(scores, rated_movie_ids, top_n, baseline=baseline),
        }

    def get_recommendations(
        self, ratings: dict, top_n: int = 10, overrides: dict = None
    ) -> dict:
        """
        End-to-end recommendations: hydrate -> forward_standard (encoder ->
        genre bottleneck) -> apply any persisted/just-submitted genre
        overrides on top of the AI-inferred bottleneck -> decode -> exclude
        already-rated movies -> translate indices back to real MovieLens
        IDs, in two parallel rankings (see _rank_both).

        With no overrides this is exactly the old "Standard Mode" behavior
        (forward_standard's own decode). With overrides, the encoder's
        genre inference is kept for every genre the user *hasn't* adjusted,
        and only the adjusted genres are shifted.

        Returns:
            {"top_rated": [...], "for_you": [...]} -- see _rank_both.
        """
        vec = hydrate_sparse_input(ratings, self.id_mapping)
        if overrides:
            _, latent_profile = self.model.forward_standard(vec)
            merged = self._merge_overrides_into_latent(latent_profile, overrides)
            predictions = self.model.forward_interactive(merged)
        else:
            predictions, _ = self.model.forward_standard(vec)

        rated_movie_ids = {int(k) for k in ratings}
        return self._rank_both(predictions[0], rated_movie_ids, top_n, self.baseline_predictions)

    def create_personalized_profile(self, ratings: dict, top_n: int = 10) -> dict:
        """
        Not available in this deployment: the real implementation (see
        main branch) does a gradient-descent fine-tune on a cloned model,
        which requires torch's autograd -- onnxruntime is inference-only.
        ai_routes.py catches this and returns 501; the frontend already
        falls back to the standard (non-personalized) profile when this
        call fails (RatingsPage.jsx's handleSave), so callers degrade
        gracefully rather than breaking.
        """
        raise NotImplementedError(
            "Profile personalization requires gradient-based fine-tuning, "
            "which this ONNX-only deployment does not support. Available "
            "in local/main-branch builds with the full PyTorch stack."
        )

    def explain_movie(self, movie_id: int, ratings: dict, overrides: dict = None) -> dict:
        """
        Human-Centered XAI for a single recommended movie:
          1. Hydrates the user's sparse ratings and runs forward_standard
             to get the latent genre profile, then applies any persisted
             genre overrides so the rationale reflects the profile the user
             is actually seeing (see get_profile), not the pre-edit one.
          2. Computes local feature importance restricted to the user's
             non-zero ratings only (never the full movie catalog) via
             compute_local_feature_importance -- against the raw,
             un-overridden signal, since this explains the AI's genuine
             rating-driven inference, not the user's own manual adjustment.
          3. Generates the natural-language soft rationale string via
             generate_soft_rationale, using the override-adjusted profile.

        Args:
            movie_id: the real MovieLens movieId being explained (NOT a
                       dense index -- this is the boundary where the API's
                       real-world ID gets translated inward).
            ratings: the user's sparse {movieId: rating} dict.
            overrides: persisted {genre_name: delta} adjustments, if any.

        Returns:
            {
              "movie_id": int,
              "title": str,
              "rationale": str,
              "feature_importance": [{"movie_id": int, "title": str, "importance": float}, ...]
            }

        Raises:
            ValueError: if movie_id is not present in the ID mapping, since
                        there is no dense index to explain against.
        """
        target_idx = self.id_mapping.movie_id_to_dense(movie_id)
        if target_idx is None:
            raise ValueError(f"movie_id {movie_id} is not present in the ID mapping.")

        vec = hydrate_sparse_input(ratings, self.id_mapping)
        _, latent_profile = self.model.forward_standard(vec)
        latent_profile = self._merge_overrides_into_latent(latent_profile, overrides)

        # Decode the (possibly override-adjusted) latent profile to get this
        # user's actual predicted score for the target movie -- the same
        # quantity _top_n_from_scores ranks with -- so the rationale's
        # lift-based fallback (see generate_soft_rationale) describes the
        # real reason this movie could have surfaced, not a re-derived
        # approximation.
        predictions = self.model.forward_interactive(latent_profile)
        predicted_score = float(predictions[0, target_idx])
        baseline_score = float(self.baseline_predictions[target_idx])

        importances_by_idx = compute_local_feature_importance(self.model, vec, target_idx)
        feature_importance = [
            {
                "movie_id": self.id_mapping.dense_to_movie_id(idx),
                "title": self.id_mapping.title_for_dense(idx),
                "importance": round(float(value), 4),
            }
            for idx, value in importances_by_idx.items()
        ]

        genre_mask_row = self.id_mapping.genre_mask[target_idx]
        rationale = generate_soft_rationale(
            sparse_input_vector=vec,
            latent_profile=latent_profile[0],
            target_movie_idx=target_idx,
            id_mapping=self.id_mapping,
            genre_mask_row=genre_mask_row,
            predicted_score=predicted_score,
            baseline_score=baseline_score,
        )

        return {
            "movie_id": movie_id,
            "title": self.id_mapping.title_for_dense(target_idx),
            "rationale": rationale,
            "feature_importance": feature_importance,
        }


ai_service = AIService()
