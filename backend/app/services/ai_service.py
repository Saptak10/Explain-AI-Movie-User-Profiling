"""
ai_service.py
--------------
Singleton service bridging FastAPI routes to the real app/ai modules:
id_mapping.py, data_pipeline.py, model.py, losses.py, xai.py.

This replaces the previous AIService implementation, which was written
against an imagined module (SoftRegularizedHCAIAutoEncoder, GENRES,
build_override_tensor, etc.) that was never actually built. The class
name, the singleton instance name (`ai_service`), and every public method
name are preserved so that main.py and ai_routes.py keep working with
only an import-path change for GENRES (now `ai_service.genres`, since the
real genre vocabulary is discovered dynamically from movies.csv rather
than hardcoded).

Training no longer builds a dense (num_users, num_movies) ratings matrix
in RAM -- the old `train_and_save` did exactly that with `np.zeros(...)`
and `iterrows()`, which is the dense-matrix approach the ml-latest
migration was specifically built to eliminate. Training now streams
ratings.csv user-by-user via SparseUserVectorDataset / DataLoader and
calls train_step from losses.py per mini-batch, so peak RAM stays
O(batch_size * num_movies) regardless of the 33M-row dataset size.
"""

import copy
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.ai.id_mapping import IdMapping, build_id_mapping
from app.ai.model import DualModeHCAIAutoEncoder
from app.ai.losses import train_step
from app.ai.data_pipeline import SparseUserVectorDataset
from app.ai.xai import (
    hydrate_sparse_input,
    compute_local_feature_importance,
    compute_genre_feature_importance,
    generate_soft_rationale,
)


class AIService:
    def __init__(self):
        self.model: DualModeHCAIAutoEncoder = None
        self.id_mapping: IdMapping = None
        self.popular_movies: list = []
        self.global_importance: dict = {}
        # Genre-neutral baseline prediction per movie -- see _init(). Used to
        # rank recommendations by personalization "lift" instead of raw
        # score, so movies with a strong genre-independent decoder bias
        # (documentaries were the clearest case: several ranked in the
        # global top-20 even against a fully neutral 50%-every-genre input)
        # don't dominate every user's recommendations regardless of fit.
        self.baseline_predictions: torch.Tensor = None
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._save_path: str = ""
        self._movies_csv_path: str = ""
        self._ratings_csv_path: str = ""

    def setup(
        self,
        model_save_path: str,
        movies_csv_path: str,
        ratings_csv_path: str,
    ) -> None:
        self._save_path = model_save_path
        self._movies_csv_path = movies_csv_path
        self._ratings_csv_path = ratings_csv_path

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

    # ── Training ─────────────────────────────────────────────────────────────

    def train_and_save(self) -> None:
        """
        Streaming training loop over the real ml-latest CSVs. Never builds
        a dense ratings matrix: SparseUserVectorDataset yields one user's
        dense vector at a time, and DataLoader only ever holds
        train_batch_size such vectors in RAM simultaneously.
        """
        from app.config.config import settings

        print("Building ID mapping from movies.csv…")
        id_mapping = build_id_mapping(self._movies_csv_path)
        print(f"  {id_mapping.num_movies} movies, {id_mapping.num_genres} genres discovered.")

        print(f"Training on device: {self.device}")
        model = DualModeHCAIAutoEncoder(id_mapping, hidden_dim=128).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=settings.train_lr, weight_decay=1e-5
        )

        dataset = SparseUserVectorDataset(
            self._ratings_csv_path,
            id_mapping,
            mask_fraction=settings.train_mask_fraction,
        )

        print(f"Training ({settings.train_epochs} epochs, streaming from ratings.csv)…")
        model.train()
        for epoch in range(settings.train_epochs):
            loader = DataLoader(dataset, batch_size=settings.train_batch_size)
            running_loss = 0.0
            num_batches = 0
            for batch in loader:
                total_loss, _, _ = train_step(
                    model,
                    optimizer,
                    batch["input_vector"].to(self.device),
                    batch["target_vector"].to(self.device),
                    batch["hidden_mask"].to(self.device),
                    lambda_reg=settings.train_lambda_reg,
                )
                running_loss += total_loss
                num_batches += 1
            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg_loss = running_loss / max(num_batches, 1)
                print(f"  Epoch {epoch + 1}/{settings.train_epochs}  avg_loss={avg_loss:.4f}")
        model.eval()

        print("Saving checkpoint…")
        Path(self._save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "movie_id_to_idx": id_mapping.movie_id_to_idx,
                "idx_to_movie_id": id_mapping.idx_to_movie_id,
                "idx_to_title": id_mapping.idx_to_title,
                "genres": id_mapping.genres,
                "genre_to_idx": id_mapping.genre_to_idx,
                "num_movies": id_mapping.num_movies,
                "num_genres": id_mapping.num_genres,
                "genre_mask": id_mapping.genre_mask,
                "hidden_dim": model.hidden_dim,
            },
            self._save_path,
        )
        self._init(model, id_mapping)

    def load(self) -> None:
        """
        Loads a saved checkpoint. The IdMapping is reconstructed from the
        checkpoint's saved dicts (not by re-reading movies.csv), so a
        loaded model always serves exactly the ID space it was trained on
        even if movies.csv has since changed on disk.
        """
        # Checkpoints may have been trained on a CUDA machine; remap to
        # whatever device is actually available here (CPU on a laptop or
        # a free-tier deploy, CUDA if present) rather than assuming either.
        ck = torch.load(self._save_path, map_location=self.device, weights_only=False)

        id_mapping = IdMapping(
            movie_id_to_idx=ck["movie_id_to_idx"],
            idx_to_movie_id=ck["idx_to_movie_id"],
            idx_to_title=ck["idx_to_title"],
            genres=ck["genres"],
            genre_to_idx=ck["genre_to_idx"],
            num_movies=ck["num_movies"],
            num_genres=ck["num_genres"],
            genre_mask=ck["genre_mask"],
        )

        model = DualModeHCAIAutoEncoder(id_mapping, hidden_dim=ck["hidden_dim"])
        model.load_state_dict(ck["state_dict"])
        model.eval()

        self._init(model, id_mapping)

    def _init(self, model: DualModeHCAIAutoEncoder, id_mapping: IdMapping) -> None:
        # load() constructs a fresh (CPU-default) model and copies checkpoint
        # values into it via load_state_dict, which preserves the destination
        # tensors' device rather than adopting the source's -- so this .to()
        # is what actually puts the model on the GPU for serving, not the
        # map_location passed to torch.load in load(). train_and_save()
        # already trains on self.device, so this is a no-op there.
        model = model.to(self.device)
        self.model = model
        self.id_mapping = id_mapping

        # Popular-movies and global-importance summaries are computed from
        # the model's learned weights only -- no dense ratings matrix is
        # read or held here, unlike the previous implementation which kept
        # self.R_tensor and self.all_latent_profiles in memory permanently.
        with torch.no_grad():
            eff = model.encoder_l2.weight @ model.encoder_l1.weight
            imp = eff.abs().mean(dim=1)
        self.global_importance = {
            g: float(imp[i].item()) for i, g in enumerate(id_mapping.genres)
        }

        # "Popular" here means highest summed genre-prior weight magnitude
        # in the decoder's output layer -- a cheap, matrix-free proxy that
        # does not require scanning ratings.csv at startup. If true
        # popularity-by-rating-count is needed, compute and cache it once
        # during train_and_save and persist it in the checkpoint instead.
        #
        # Cached pool is much larger than any one page shown to a user
        # (get_popular_sample below draws a random subset per request) --
        # this is what lets the Rate page's "Refresh Suggestions" surface a
        # genuinely different batch instead of the same fixed 50 every
        # time, for users whose first batch didn't include enough movies
        # they've actually seen to rate.
        with torch.no_grad():
            decoder_eff = model.decoder_l2.weight @ model.decoder_l1.weight
            movie_salience = decoder_eff.abs().sum(dim=1)
        top_idx = torch.argsort(movie_salience, descending=True)[:500].tolist()
        self.popular_movies = [
            {
                "id": id_mapping.dense_to_movie_id(i),
                "title": id_mapping.title_for_dense(i),
            }
            for i in top_idx
        ]

        self.baseline_predictions = self._compute_baseline(model)
        print("AI service ready.")

    def _compute_baseline(self, model: DualModeHCAIAutoEncoder) -> torch.Tensor:
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
        neutral = torch.full((1, model.num_genres), 0.5, device=self.device)
        model.eval()
        with torch.no_grad():
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

    # ── Inference ────────────────────────────────────────────────────────────

    def _merge_overrides_into_latent(
        self, latent_profile: torch.Tensor, overrides: dict
    ) -> torch.Tensor:
        """
        Applies persisted/just-submitted genre-preference deltas on top of
        the AI-inferred (1, num_genres) latent profile, clamped back into
        [0, 1]. Returns a new tensor -- never mutates latent_profile in
        place, since callers may still need the un-overridden version.
        """
        if not overrides:
            return latent_profile
        merged = latent_profile.clone()
        for genre_name, delta in overrides.items():
            genre_idx = self.id_mapping.genre_to_idx.get(genre_name)
            if genre_idx is not None:
                merged[0, genre_idx] = torch.clamp(
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
        vec = hydrate_sparse_input(ratings, self.id_mapping).to(self.device)
        self.model.eval()
        with torch.no_grad():
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

        Kept as a separate, more expensive endpoint from GET /api/profile
        (which stays cheap, a single forward pass) since this costs O(k)
        extra forward passes per genre explained, k = the user's number of
        non-zero ratings -- with ~19 genres and typically tens of ratings,
        explaining every genre is still comfortably real-time; this is why
        it's fetched lazily by the frontend on first "Why?" click, not on
        every page load.

        Returns:
            {
              "profile": {genre_name: percentage, ...},
              "genre_explanations": {
                genre_name: [{"movie_id": int, "title": str, "importance": float}, ...],
                ...
              }
            }
        """
        vec = hydrate_sparse_input(ratings, self.id_mapping).to(self.device)
        self.model.eval()
        with torch.no_grad():
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
        scores: torch.Tensor,
        rated_movie_ids: set,
        top_n: int,
        baseline: torch.Tensor = None,
    ) -> list:
        """
        Shared candidate-ranking tail used by every inference path that ends
        in "top_n scored movies, excluding ones the user already rated":
        get_recommendations and create_personalized_profile both call this
        instead of duplicating the same argsort/exclude/translate loop.

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
            rank_scores = torch.where(passes_floor, lift, scores - 100.0)
        candidate_order = torch.argsort(rank_scores, descending=True)

        results = []
        for dense_idx_tensor in candidate_order:
            dense_idx = int(dense_idx_tensor.item())
            real_movie_id = self.id_mapping.dense_to_movie_id(dense_idx)
            if real_movie_id in rated_movie_ids:
                continue
            results.append({
                "movie_id": real_movie_id,
                "title": self.id_mapping.title_for_dense(dense_idx),
                "score": round(float(scores[dense_idx].item()), 3),
            })
            if len(results) >= top_n:
                break
        return results

    def _rank_both(
        self, scores: torch.Tensor, rated_movie_ids: set, top_n: int, baseline: torch.Tensor
    ) -> dict:
        """
        Produces both recommendation orderings from a single, already-computed
        predictions tensor -- no extra model forward pass needed, just two
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
        and only the adjusted genres are shifted -- this replaces the old,
        separate get_recommendations_from_profile (which fed a hand-built
        override vector into forward_interactive standalone, bypassing the
        encoder for the *entire* profile, not just the overridden genres).

        Returns:
            {"top_rated": [...], "for_you": [...]} -- see _rank_both.
        """
        vec = hydrate_sparse_input(ratings, self.id_mapping).to(self.device)
        self.model.eval()
        with torch.no_grad():
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
        One-time personalization fine-tune, run right after a user builds
        their profile ("Build My Profile" on the Rate page).

        Never touches self.model: ai_service.model is a single instance
        shared across every concurrent request's inference calls (via
        asyncio.to_thread, so genuinely concurrent threads), with no lock.
        Fine-tuning in place would race with other users' inference and
        would leak this user's ratings into everyone else's recommendations.
        Instead this clones the model, fine-tunes only the clone against
        this user's own known ratings (no held-out split -- the goal is a
        personalization nudge, not a generalization eval), computes the
        profile + recommendations from the clone, and lets the clone and its
        optimizer be garbage-collected when this method returns.
        """
        from app.config.config import settings

        clone = copy.deepcopy(self.model)
        clone.train()
        optimizer = torch.optim.Adam(
            clone.parameters(), lr=settings.personalize_lr, weight_decay=1e-5
        )

        vec = hydrate_sparse_input(ratings, self.id_mapping).to(self.device)
        known_mask = vec != 0

        for _ in range(settings.personalize_epochs):
            train_step(
                clone,
                optimizer,
                vec,
                vec,
                known_mask,
                lambda_reg=settings.train_lambda_reg,
            )

        clone.eval()
        with torch.no_grad():
            predictions, latent_profile = clone.forward_standard(vec)
        clone_baseline = self._compute_baseline(clone)

        profile = clone.extract_taste_profile(latent_profile[0], self.id_mapping.genres)
        rated_movie_ids = {int(k) for k in ratings}
        recommendations = self._top_n_from_scores(
            predictions[0], rated_movie_ids, top_n, baseline=clone_baseline
        )

        return {"profile": profile, "recommendations": recommendations}

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

        vec = hydrate_sparse_input(ratings, self.id_mapping).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, latent_profile = self.model.forward_standard(vec)
        latent_profile = self._merge_overrides_into_latent(latent_profile, overrides)

        # Decode the (possibly override-adjusted) latent profile to get this
        # user's actual predicted score for the target movie -- the same
        # quantity _top_n_from_scores ranks with -- so the rationale's
        # lift-based fallback (see generate_soft_rationale) describes the
        # real reason this movie could have surfaced, not a re-derived
        # approximation.
        with torch.no_grad():
            predictions = self.model.forward_interactive(latent_profile)
        predicted_score = float(predictions[0, target_idx].item())
        baseline_score = float(self.baseline_predictions[target_idx].item())

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
