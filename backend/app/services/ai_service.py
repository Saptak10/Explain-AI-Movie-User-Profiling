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
    generate_soft_rationale,
)


class AIService:
    def __init__(self):
        self.model: DualModeHCAIAutoEncoder = None
        self.id_mapping: IdMapping = None
        self.popular_movies: list = []
        self.global_importance: dict = {}
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

        model = DualModeHCAIAutoEncoder(id_mapping, hidden_dim=128)
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
                    batch["input_vector"],
                    batch["target_vector"],
                    batch["hidden_mask"],
                    lambda_reg=settings.train_lambda_reg,
                    epsilon_clip=settings.train_epsilon_clip,
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
        ck = torch.load(self._save_path, weights_only=False)

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
        with torch.no_grad():
            decoder_eff = model.decoder_l2.weight @ model.decoder_l1.weight
            movie_salience = decoder_eff.abs().sum(dim=1)
        top_idx = torch.argsort(movie_salience, descending=True)[:50].tolist()
        self.popular_movies = [
            {
                "id": id_mapping.dense_to_movie_id(i),
                "title": id_mapping.title_for_dense(i),
            }
            for i in top_idx
        ]
        print("AI service ready.")

    # ── Inference ────────────────────────────────────────────────────────────

    def get_profile(self, ratings: dict) -> dict:
        """
        Standard-mode genre taste profile: hydrates the sparse ratings dict,
        runs the full encoder -> bottleneck pass, and returns a
        {genre_name: percentage} dict via extract_taste_profile.
        """
        vec = hydrate_sparse_input(ratings, self.id_mapping)
        self.model.eval()
        with torch.no_grad():
            _, latent_profile = self.model.forward_standard(vec)
        return self.model.extract_taste_profile(latent_profile[0], self.id_mapping.genres)

    def get_recommendations(self, ratings: dict, top_n: int = 10) -> list:
        """
        Standard Mode end-to-end recommendations: hydrate -> forward_standard
        -> exclude already-rated movies -> top_n by score -> translate
        indices back to real MovieLens IDs.
        """
        vec = hydrate_sparse_input(ratings, self.id_mapping)
        self.model.eval()
        with torch.no_grad():
            predictions, _ = self.model.forward_standard(vec)

        scores = predictions[0]
        rated_movie_ids = {int(k) for k in ratings}
        candidate_order = torch.argsort(scores, descending=True)

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

    def get_recommendations_from_profile(
        self, genre_overrides: dict, ratings: dict, top_n: int = 10
    ) -> list:
        """
        Interactive Profile Mode: accepts {genre_name: weight} slider
        overrides from the UI, builds the (1, num_genres) override tensor,
        runs forward_interactive (which never touches the encoder layers),
        and returns top_n recommendations the same way as standard mode.
        """
        override_vec = torch.zeros(1, self.id_mapping.num_genres)
        for genre_name, weight in genre_overrides.items():
            genre_idx = self.id_mapping.genre_to_idx.get(genre_name)
            if genre_idx is not None:
                override_vec[0, genre_idx] = float(weight)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model.forward_interactive(override_vec)

        scores = predictions[0]
        rated_movie_ids = {int(k) for k in ratings}
        candidate_order = torch.argsort(scores, descending=True)

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

    def explain_movie(self, movie_id: int, ratings: dict) -> dict:
        """
        Human-Centered XAI for a single recommended movie:
          1. Hydrates the user's sparse ratings and runs forward_standard
             to get the latent genre profile.
          2. Computes local feature importance restricted to the user's
             non-zero ratings only (never the full movie catalog) via
             compute_local_feature_importance.
          3. Generates the natural-language soft rationale string via
             generate_soft_rationale.

        Args:
            movie_id: the real MovieLens movieId being explained (NOT a
                       dense index -- this is the boundary where the API's
                       real-world ID gets translated inward).
            ratings: the user's sparse {movieId: rating} dict.

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
        self.model.eval()
        with torch.no_grad():
            _, latent_profile = self.model.forward_standard(vec)

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
        )

        return {
            "movie_id": movie_id,
            "title": self.id_mapping.title_for_dense(target_idx),
            "rationale": rationale,
            "feature_importance": feature_importance,
        }


ai_service = AIService()
