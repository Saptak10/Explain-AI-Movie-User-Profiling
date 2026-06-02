import numpy as np
import torch
from pathlib import Path

from app.ai.ai import (
    GENRES,
    SoftRegularizedHCAIAutoEncoder,
    build_override_tensor,
    explain_with_lime,
    generate_soft_xai_explanation,
    load_movielens_data,
    train_step,
)


class AIService:
    def __init__(self):
        self.model: SoftRegularizedHCAIAutoEncoder = None
        self.movie_id_to_idx: dict = {}
        self.idx_to_title: dict = {}
        self.num_movies: int = 0
        self.num_genres: int = len(GENRES)
        self.R_tensor: torch.Tensor = None
        self.all_latent_profiles: torch.Tensor = None
        self.popular_movies: list = []
        self.global_importance: dict = {}
        self._save_path: str = ""

    def setup(self, save_path: str) -> None:
        self._save_path = save_path

    # ── Training ─────────────────────────────────────────────────────────────

    def train_and_save(self) -> None:
        import torch.optim as optim

        print("Loading MovieLens data…")
        ratings_df, movies_df = load_movielens_data()

        movie_id_to_idx = {mid: i for i, mid in enumerate(movies_df["movieId"].unique())} # Map movie IDs to indices
        idx_to_title = { # Map indices back to movie titles
            movie_id_to_idx[int(r["movieId"])]: r["title"]
            for _, r in movies_df.iterrows()
            if int(r["movieId"]) in movie_id_to_idx
        }
        user_id_to_idx = {uid: i for i, uid in enumerate(ratings_df["userId"].unique())} # Map user IDs to indices
        num_movies = len(movie_id_to_idx)

        print(f"Building matrices ({len(user_id_to_idx)} users × {num_movies} movies)…")
        R = np.zeros((len(user_id_to_idx), num_movies), dtype=np.float32)
        for _, r in ratings_df.iterrows():
            R[user_id_to_idx[int(r["userId"])], movie_id_to_idx[int(r["movieId"])]] = float(r["rating"])

        genre_np = np.zeros((num_movies, len(GENRES)), dtype=np.float32)
        for _, r in movies_df.iterrows():
            m = movie_id_to_idx[int(r["movieId"])]
            for g in r["genres"].split("|"):
                if g in GENRES:
                    genre_np[m, GENRES.index(g)] = 1.0

        model = SoftRegularizedHCAIAutoEncoder(num_movies, len(GENRES), genre_np) # Initialize model
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        R_tensor = torch.tensor(R, dtype=torch.float32)
        mask = R_tensor > 0.0

        print("Training (25 epochs)…")
        model.train()
        for epoch in range(25):
            perm = torch.randperm(R_tensor.size(0))
            for i in range(0, R_tensor.size(0), 32):
                idx = perm[i: i + 32]
                train_step(model, optimizer, R_tensor[idx], mask[idx]) # Train on mini-batch
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/25 done")
        model.eval()

        print("Saving checkpoint…")
        Path(self._save_path).parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        torch.save(     # Save model state and metadata for later loading
            {
                "state_dict": model.state_dict(),
                "movie_id_to_idx": movie_id_to_idx,
                "idx_to_title": idx_to_title,
                "num_movies": num_movies,
                "genre_np": genre_np,
                "R_matrix": R,
            },
            self._save_path,
        )
        self._init(model, movie_id_to_idx, idx_to_title, num_movies, R_tensor)

    def load(self) -> None:     # Load model state and metadata from saved checkpoint
        ck = torch.load(self._save_path, weights_only=False)
        model = SoftRegularizedHCAIAutoEncoder(ck["num_movies"], len(GENRES), ck["genre_np"])
        model.load_state_dict(ck["state_dict"])
        model.eval()
        R_tensor = torch.tensor(ck["R_matrix"], dtype=torch.float32)
        self._init(model, ck["movie_id_to_idx"], ck["idx_to_title"], ck["num_movies"], R_tensor)

    def _init(self, model, movie_id_to_idx, idx_to_title, num_movies, R_tensor) -> None:
        self.model = model
        self.movie_id_to_idx = movie_id_to_idx
        self.idx_to_title = idx_to_title
        self.num_movies = num_movies
        self.R_tensor = R_tensor

        with torch.no_grad():
            _, self.all_latent_profiles = self.model(R_tensor)

        counts = (R_tensor > 0).sum(dim=0).numpy()
        top_idx = np.argsort(counts)[::-1][:50]
        self.popular_movies = [
            {"id": int(i), "title": idx_to_title.get(int(i), f"Movie #{i}")}
            for i in top_idx
        ]

        with torch.no_grad():
            eff = model.encoder_l2.weight @ model.encoder_l1.weight
            imp = eff.abs().mean(dim=1).cpu().numpy()
        self.global_importance = {g: float(imp[i]) for i, g in enumerate(GENRES)}
        print("AI service ready.")

    # ── Inference ────────────────────────────────────────────────────────────

    def _user_vec(self, ratings: dict) -> torch.Tensor:
        vec = torch.zeros(1, self.num_movies)
        for mid, rating in ratings.items():
            idx = int(mid)
            if 0 <= idx < self.num_movies:
                vec[0, idx] = float(rating)
        return vec

    def get_profile(self, ratings: dict) -> dict:
        vec = self._user_vec(ratings)
        with torch.no_grad():
            _, latent = self.model(vec)
        return {g: float(latent[0][i].item()) for i, g in enumerate(GENRES)}

    def get_recommendations(
        self,
        ratings: dict,
        top_n: int = 10,
        overrides: dict = None,
        alpha: float = 3.0,
    ) -> list:
        vec = self._user_vec(ratings)
        override_t = build_override_tensor(overrides, self.num_genres) if overrides else None
        with torch.no_grad():
            output, _ = self.model(vec, user_overrides=override_t, alpha=alpha)
        scores = output.squeeze().numpy()
        rated = {int(k) for k in ratings}
        results = []
        for idx in np.argsort(scores)[::-1]:
            if int(idx) not in rated and len(results) < top_n:
                results.append({
                    "movie_id": int(idx),
                    "title": self.idx_to_title.get(int(idx), f"Movie #{idx}"),
                    "score": round(float(scores[idx]), 3),
                })
        return results

    def get_recommendations_from_profile(
        self, edited: dict, ratings: dict, top_n: int = 10
    ) -> list:
        pt = torch.zeros(1, self.num_genres)
        for g, v in edited.items():
            if g in GENRES:
                pt[0, GENRES.index(g)] = float(v)
        with torch.no_grad():
            output = self.model.recommend_from_edited_profile(pt)
        scores = output.squeeze().numpy()
        rated = {int(k) for k in ratings}
        results = []
        for idx in np.argsort(scores)[::-1]:
            if int(idx) not in rated and len(results) < top_n:
                results.append({
                    "movie_id": int(idx),
                    "title": self.idx_to_title.get(int(idx), f"Movie #{idx}"),
                    "score": round(float(scores[idx]), 3),
                })
        return results

    def explain_movie(self, movie_idx: int, ratings: dict, method: str = "soft") -> dict:
        vec = self._user_vec(ratings)
        with torch.no_grad():
            _, latent = self.model(vec)

        if method == "lime":
            result = explain_with_lime(self.model, latent[0].unsqueeze(0), movie_idx, n_samples=200)
            if result:
                return {
                    "method": "lime",
                    "contributions": [{"genre": g, "value": float(v)} for g, v in result],
                }
            return {"method": "lime", "contributions": []}

        title = self.idx_to_title.get(movie_idx, f"Movie #{movie_idx}")
        text = generate_soft_xai_explanation(title, movie_idx, self.model, latent[0])
        with torch.no_grad():
            eff = self.model.encoder_l2.weight @ self.model.encoder_l1.weight
            weights = eff[:, movie_idx].cpu().numpy()
            profile = latent[0].cpu().numpy()
        contributions = sorted(
            [{"genre": GENRES[i], "value": float(weights[i] * profile[i])} for i in range(len(GENRES))],
            key=lambda x: abs(x["value"]),
            reverse=True,
        )
        return {"method": "soft", "text": text, "contributions": contributions}


ai_service = AIService()
