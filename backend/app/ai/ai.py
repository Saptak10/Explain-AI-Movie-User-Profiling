"""
Main file containing ML/AI logic for Transparent & Editable User Profiling.
Architecture: Soft-Regularized HCAI Autoencoder with Hybrid Override Path
Dataset: MovieLens Latest Small (100k ratings)
"""
 
# region imports and config
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
 
# Optional XAI dependencies - graceful degradation if not installed.
# Install via: pip install shap scikit-learn
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
 
try:
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
 
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(current_dir, "..", "database", "ml-latest-small"))
 
OVERRIDE_MAP = {
    "strongly_reduce":  -2.0,
    "slightly_reduce":  -1.0,
    "neutral":           0.0,
    "slightly_increase": 1.0,
    "strongly_increase": 2.0,
}
 
GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
# endregion
 
 
# region dataset helpers
def load_movielens_data():
    """Load the MovieLens CSV files from DATA_DIR."""
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"), encoding="utf-8")
    movies  = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"),  encoding="utf-8")
    return ratings, movies
 
 
def build_pure_genre_matrix(movies_df):
    """
    Build the pure genre matrix M_genre.
    Shape: (num_movies x 18), binary encoding of genre membership.
    """
    matrix = pd.DataFrame(0.0, index=movies_df["movieId"], columns=GENRES)
    for _, row in movies_df.iterrows():
        for g in row["genres"].split("|"):
            if g in GENRES:
                matrix.at[row["movieId"], g] = 1.0
    return matrix
# endregion
 
 
# region hcai utility functions
def build_override_tensor(user_category_dict: dict, num_genres: int) -> torch.Tensor:
    """
    Convert a human-readable genre override dict into a float tensor for model.forward().
 
    Maps string keys from OVERRIDE_MAP to their float values and places them
    at the correct genre index. Unspecified genres default to 0.0 (neutral).
 
    Args:
        user_category_dict: {genre_name: override_key} e.g. {"Sci-Fi": "strongly_increase"}
        num_genres:         Number of genres (len(GENRES))
 
    Returns:
        Float tensor of shape (1, num_genres).
    """
    tensor = torch.zeros(1, num_genres, dtype=torch.float32)
    for genre, category in user_category_dict.items():
        if genre in GENRES and category in OVERRIDE_MAP:
            tensor[0, GENRES.index(genre)] = OVERRIDE_MAP[category]
    return tensor
# endregion
 
 
# region neural network model
class SoftRegularizedHCAIAutoEncoder(nn.Module):
    def __init__(self, num_movies, num_genres, pure_genre_matrix_np, hidden_dim=128):
        super().__init__()
        self.num_movies = num_movies
        self.num_genres = num_genres
        self.hidden_dim = hidden_dim
 
        # Register the pure genre knowledge base as a fixed buffer
        target_tensor = torch.tensor(pure_genre_matrix_np, dtype=torch.float32).T
        self.register_buffer("target_genre_matrix", target_tensor)
 
        # Encoder
        self.encoder_l1  = nn.Linear(num_movies, hidden_dim, bias=False)
        self.encoder_act = nn.ReLU()
        self.encoder_l2  = nn.Linear(hidden_dim, num_genres, bias=False)
 
        # Decoder
        self.dropout     = nn.Dropout(p=0.4)
        self.decoder_l1  = nn.Linear(num_genres, hidden_dim)
        self.decoder_act = nn.ReLU()
        self.decoder_l2  = nn.Linear(hidden_dim, num_movies)
 
        # Knowledge injection: initialize encoder weights from genre matrix
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder_l1.weight, a=math.sqrt(5))
            self.encoder_l1.weight[:num_genres, :].copy_(target_tensor)
            nn.init.zeros_(self.encoder_l2.weight)
            self.encoder_l2.weight[:, :num_genres].copy_(torch.eye(num_genres))
 
    def forward(self, x, user_overrides=None, alpha=1.0):
        """
        Forward pass with optional hybrid override.
 
        Path A: standard autoencoder prediction (always active).
        Path B: adds a genre-weighted bias from user_overrides (optional).
 
        Args:
            x:              User rating vector (batch_size x num_movies).
            user_overrides: Float tensor (1 x num_genres) from build_override_tensor().
            alpha:          Scaling factor for the override bias.
 
        Returns:
            (output scores, latent genre profile)
        """
        # Path A: autoencoder
        h_enc = self.encoder_act(self.encoder_l1(x))
        latent_profile = torch.sigmoid(self.encoder_l2(h_enc))
 
        h_dec      = self.decoder_act(self.decoder_l1(self.dropout(latent_profile)))
        ann_output = torch.sigmoid(self.decoder_l2(h_dec)) * 5.0
 
        # Path B: hybrid override
        if user_overrides is not None:
            explicit_bias = torch.matmul(user_overrides, self.target_genre_matrix)
            return ann_output + (alpha * explicit_bias), latent_profile
 
        return ann_output, latent_profile
 
    def apply_weight_clipping(self, epsilon=0.15):
        """Clip encoder weights so they don't drift too far from the genre-matrix prior."""
        with torch.no_grad():
            dev_l1 = self.encoder_l1.weight[:self.num_genres, :] - self.target_genre_matrix
            self.encoder_l1.weight[:self.num_genres, :].copy_(
                self.target_genre_matrix + torch.clamp(dev_l1, -epsilon, epsilon)
            )
            identity = torch.eye(self.num_genres, device=self.encoder_l2.weight.device)
            dev_l2 = self.encoder_l2.weight[:, :self.num_genres] - identity
            self.encoder_l2.weight[:, :self.num_genres].copy_(
                identity + torch.clamp(dev_l2, -epsilon, epsilon)
            )
 
    def recommend_from_edited_profile(self, edited_profile: torch.Tensor) -> torch.Tensor:
        """
        Decode a manually edited genre profile directly into movie scores.
 
        Bypasses the encoder entirely - the user's profile is passed straight
        to the decoder, which acts as a translator from genre space to film scores.
 
        Args:
            edited_profile: Tensor (1 x num_genres), values in [0, 1].
 
        Returns:
            Movie scores (1 x num_movies) in [0, 5].
        """
        h_dec = self.decoder_act(self.decoder_l1(edited_profile))
        return torch.sigmoid(self.decoder_l2(h_dec)) * 5.0
 
    def explain_override_impact(
        self,
        ann_output:    torch.Tensor,
        final_output:  torch.Tensor,
        idx_to_title:  dict,
        top_n:         int = 5,
    ) -> dict:
        """
        Show which movies were most affected by a user override.
 
        Computes the score delta between the pre- and post-override outputs
        and returns the top boosted and suppressed films.
 
        Args:
            ann_output:   Scores before override (1 x num_movies).
            final_output: Scores after override  (1 x num_movies).
            idx_to_title: Mapping from dense index to movie title.
            top_n:        Number of top/bottom films to return.
 
        Returns:
            Dict with keys "boosted" and "suppressed", each a list of (title, delta).
        """
        delta = (final_output - ann_output).squeeze()
 
        top_up_vals,   top_up_idx   = torch.topk( delta, top_n)
        top_down_vals, top_down_idx = torch.topk(-delta, top_n)
 
        boosted = [
            (idx_to_title.get(i.item(), f"Film #{i.item()}"), v.item())
            for i, v in zip(top_up_idx, top_up_vals)
        ]
        suppressed = [
            (idx_to_title.get(i.item(), f"Film #{i.item()}"), -v.item())
            for i, v in zip(top_down_idx, top_down_vals)
        ]
        return {"boosted": boosted, "suppressed": suppressed}
 
    def get_semantic_loss(self):
        """
        Compute the semantic regularization loss (squared drift from genre-matrix prior).
 
        Two components:
          - L1 term: encoder_l1 rows should stay close to M_genre^T.
          - L2 term: encoder_l2 columns should stay close to the identity matrix.
 
        Uses torch.mean() to avoid numerical domination over the prediction loss.
        """
        loss_l1 = torch.mean(
            (self.encoder_l1.weight[:self.num_genres, :] - self.target_genre_matrix) ** 2
        )
        identity = torch.eye(self.num_genres, device=self.encoder_l2.weight.device)
        loss_l2 = torch.mean(
            (self.encoder_l2.weight[:, :self.num_genres] - identity) ** 2
        )
        return loss_l1 + loss_l2
 
 
def generate_soft_xai_explanation(movie_title, movie_idx, model, single_user_profile):
    """
    Generate a natural-language explanation for a recommendation using fuzzy thresholding.
 
    Two tiers:
      - Core genres:   effective_weight > 0.85 AND user affinity > 0.60
      - Subtle genres: 0.20 < effective_weight <= 0.85
 
    Effective weights = encoder_l2.weight @ encoder_l1.weight  (18 x num_movies).
 
    Args:
        movie_title:         Title string for the recommended film.
        movie_idx:           Dense tensor index of the film.
        model:               Trained SoftRegularizedHCAIAutoEncoder.
        single_user_profile: Latent genre profile tensor (num_genres,).
 
    Returns:
        Explanation string.
    """
    with torch.no_grad():
        eff_weights = model.encoder_l2.weight @ model.encoder_l1.weight
        weights = eff_weights[:, movie_idx].cpu().numpy()
        profile = single_user_profile.cpu().numpy().flatten()
 
    core_genres = [
        GENRES[i] for i in range(len(GENRES))
        if weights[i] > 0.85 and profile[i] > 0.6
    ]
    subtle_genres = [
        GENRES[i] for i in range(len(GENRES))
        if 0.20 < weights[i] <= 0.85 and profile[i] > 0.3
    ]
 
    if core_genres:
        explanation = (
            f'"{movie_title}" matches your strong interest in: {", ".join(core_genres)}.'
        )
    else:
        explanation = (
            f'"{movie_title}" is recommended based on general collaborative filtering patterns.'
        )
 
    if subtle_genres:
        explanation += f" Subtle genre influences detected: {', '.join(subtle_genres)}."
 
    return explanation
# endregion
 
 
# region loss and training utilities
def masked_mse_loss(predictions, targets, mask):
    """
    MSE loss computed only over movies the user has actually rated.
    Avoids penalizing the model for zero-padded (unseen) entries.
    """
    loss        = (predictions - targets) ** 2
    masked_loss = loss * mask.float()
    num_rated   = mask.float().sum()
    return masked_loss.sum() / (num_rated + 1e-8)
 
 
def train_step(model, optimizer, user_batch, mask_batch, lambda_reg=0.05, epsilon_clip=0.15):
    """
    Single training step: prediction loss + semantic regularization + weight clipping.
 
    Args:
        model:        SoftRegularizedHCAIAutoEncoder instance.
        optimizer:    PyTorch optimizer.
        user_batch:   Rating vectors for a mini-batch (batch_size x num_movies).
        mask_batch:   Boolean mask of rated entries (batch_size x num_movies).
        lambda_reg:   Weight for the semantic regularization term.
        epsilon_clip: Clipping radius for encoder weight deviation.
 
    Returns:
        (total_loss, prediction_loss, semantic_loss) as Python floats.
    """
    optimizer.zero_grad()
 
    predictions, _ = model(user_batch)
 
    pred_loss     = masked_mse_loss(predictions, user_batch, mask_batch)
    semantic_loss = model.get_semantic_loss()
    total_loss    = pred_loss + (lambda_reg * semantic_loss)
 
    total_loss.backward()
    optimizer.step()
 
    model.apply_weight_clipping(epsilon=epsilon_clip)
 
    return total_loss.item(), pred_loss.item(), semantic_loss.item()
# endregion
 
 
# region explainable ai (XAI)
 
# ---------------------------------------------------------------------------
# Method 1: SHAP (SHapley Additive exPlanations) - local, exact
# ---------------------------------------------------------------------------
 
def explain_with_shap(
    model,
    user_profile:        torch.Tensor,
    movie_idx:           int,
    background_profiles: torch.Tensor,
    n_samples:           int = 100,
) -> list | None:
    """
    KernelSHAP applied to the decoder sub-network for a specific movie.
 
    Explains how much each genre dimension in the latent profile contributed
    to the predicted score for movie_idx.
 
    Args:
        model:               Trained model (eval mode recommended).
        user_profile:        Latent genre profile (1 x num_genres).
        movie_idx:           Dense tensor index of the target film.
        background_profiles: Background sample for SHAP baseline (k x num_genres).
        n_samples:           Number of SHAP perturbation samples.
 
    Returns:
        List of (genre, shap_value) sorted by absolute influence, or None if shap
        is not installed.
    """
    if not SHAP_AVAILABLE:
        print("   shap not installed - skipped. (pip install shap)")
        return None
 
    def decoder_score_fn(profiles_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t      = torch.tensor(profiles_np, dtype=torch.float32)
            h      = model.decoder_act(model.decoder_l1(t))
            scores = torch.sigmoid(model.decoder_l2(h)) * 5.0
            return scores[:, movie_idx].numpy()
 
    background_np = background_profiles.numpy()
    explainer     = shap.KernelExplainer(decoder_score_fn, background_np)
 
    profile_np  = user_profile.numpy().reshape(1, -1)
    shap_values = explainer.shap_values(profile_np, nsamples=n_samples, silent=True)
 
    contributions = list(zip(GENRES, shap_values.flatten()))
    return sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
 
 
# ---------------------------------------------------------------------------
# Method 2: LIME (Local Interpretable Model-agnostic Explanations) - local, approximate
# ---------------------------------------------------------------------------
 
def explain_with_lime(
    model,
    user_profile: torch.Tensor,
    movie_idx:    int,
    n_samples:    int = 300,
) -> list | None:
    """
    LIME in the latent genre space (18 dims) rather than the raw film space (9742 dims).
 
    Perturbs the genre profile with Gaussian noise, scores each variant with
    the decoder, then fits a local Ridge regression to estimate feature importance.
 
    Note: LIME results can vary across runs due to random sampling. Use SHAP
    for quantitative comparisons; LIME is useful as a complementary visualization.
 
    Args:
        model:        Trained model (eval mode recommended).
        user_profile: Latent genre profile (1 x num_genres).
        movie_idx:    Dense tensor index of the target film.
        n_samples:    Number of perturbed samples (300 is sufficient for 18 features).
 
    Returns:
        List of (genre, coefficient) sorted by absolute importance, or None if
        scikit-learn is not installed.
    """
    if not SKLEARN_AVAILABLE:
        print("   scikit-learn not installed - skipped. (pip install scikit-learn)")
        return None
 
    profile_np = user_profile.numpy().flatten()
 
    np.random.seed(42)
    noise     = np.random.normal(0, 0.15, size=(n_samples, len(GENRES)))
    perturbed = np.clip(profile_np + noise, 0.0, 1.0).astype(np.float32)
 
    with torch.no_grad():
        t      = torch.tensor(perturbed, dtype=torch.float32)
        h      = model.decoder_act(model.decoder_l1(t))
        scores = (torch.sigmoid(model.decoder_l2(h)) * 5.0)[:, movie_idx].numpy()
 
    local_model = Ridge(alpha=1.0)
    local_model.fit(perturbed, scores)
 
    contributions = list(zip(GENRES, local_model.coef_))
    return sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
 
 
# ---------------------------------------------------------------------------
# Method 3: Permutation Feature Importance - global, model-agnostic
# ---------------------------------------------------------------------------
 
def _eval_decoder_mse(model, profiles: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute masked MSE of the decoder over all users (no dropout, no encoder)."""
    with torch.no_grad():
        h     = model.decoder_act(model.decoder_l1(profiles))
        preds = torch.sigmoid(model.decoder_l2(h)) * 5.0
        mask  = targets > 0.0
        return masked_mse_loss(preds, targets, mask).item()
 
 
def compute_genre_permutation_importance(
    model,
    all_profiles: torch.Tensor,
    all_targets:  torch.Tensor,
    n_repeats:    int = 5,
) -> dict:
    """
    Global permutation feature importance across all user profiles.
 
    For each genre dimension, shuffle that column across users and measure
    how much the masked MSE increases. A large increase indicates the genre
    is important for overall recommendation quality.
 
    Args:
        model:        Trained model (eval mode).
        all_profiles: Latent profiles for all users (num_users x num_genres).
        all_targets:  Ground-truth rating matrix (num_users x num_movies), 0 = unrated.
        n_repeats:    Permutation repetitions per genre (reduces variance).
 
    Returns:
        Dict {genre: mean_mse_increase}, sorted descending by importance.
    """
    baseline    = _eval_decoder_mse(model, all_profiles, all_targets)
    importances = {}
 
    for g_idx, genre in enumerate(GENRES):
        deltas = []
        for _ in range(n_repeats):
            permuted = all_profiles.clone()
            perm_order = torch.randperm(permuted.size(0))
            permuted[:, g_idx] = permuted[perm_order, g_idx]
            deltas.append(_eval_decoder_mse(model, permuted, all_targets) - baseline)
        importances[genre] = float(np.mean(deltas))
 
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
 
# endregion
 
 
# region main evaluation pipeline
 
if __name__ == "__main__":
    import torch.optim as optim
    import time
 
    print("=" * 70)
    print("STARTING MOVIELENS EVALUATION PIPELINE")
    print("=" * 70)
 
    if not os.path.exists(DATA_DIR) or not os.path.exists(os.path.join(DATA_DIR, "ratings.csv")):
        print(f"ERROR: Dataset folder '{DATA_DIR}' or CSV files not found.")
        print("Please download the ml-latest-small dataset from MovieLens and extract it here.")
        exit()
 
    print("Step 1: Loading MovieLens files...")
    ratings_df, movies_df = load_movielens_data()
    print(f"-> {len(ratings_df)} ratings and {len(movies_df)} movies loaded.")
 
    print("\nStep 2: Building index mappings...")
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movies_df["movieId"].unique())}
 
    idx_to_movie_title = {}
    for _, row in movies_df.iterrows():
        m_id = int(row["movieId"])
        if m_id in movie_id_to_idx:
            idx_to_movie_title[movie_id_to_idx[m_id]] = row["title"]
 
    user_id_to_idx = {uid: idx for idx, uid in enumerate(ratings_df["userId"].unique())}
 
    num_users  = len(user_id_to_idx)
    num_movies = len(movie_id_to_idx)
    num_genres = len(GENRES)
 
    print(f"-> Dense matrix dimensions: {num_users} users x {num_movies} movies")
 
    print("\nStep 3: Building rating matrix and genre matrix...")
    R_matrix = np.zeros((num_users, num_movies), dtype=np.float32)
    for _, row in ratings_df.iterrows():
        u_idx = user_id_to_idx[int(row["userId"])]
        m_idx = movie_id_to_idx[int(row["movieId"])]
        R_matrix[u_idx, m_idx] = float(row["rating"])
 
    pure_genre_matrix_np = np.zeros((num_movies, num_genres), dtype=np.float32)
    for _, row in movies_df.iterrows():
        m_idx = movie_id_to_idx[int(row["movieId"])]
        for g in row["genres"].split("|"):
            if g in GENRES:
                pure_genre_matrix_np[m_idx, GENRES.index(g)] = 1.0
 
    sparsity = (1.0 - (np.count_nonzero(R_matrix) / R_matrix.size)) * 100
    print(f"-> Dataset sparsity: {sparsity:.2f}%")
 
    print("\nStep 4: Initializing model...")
    model     = SoftRegularizedHCAIAutoEncoder(num_movies, num_genres, pure_genre_matrix_np)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    print(f"-> Encoder initialized with genre matrix of shape {pure_genre_matrix_np.shape}.")
 
    print("\nStep 5: Training over all users...")
    R_tensor              = torch.tensor(R_matrix, dtype=torch.float32)
    global_evaluation_mask = R_tensor > 0.0
 
    EPOCHS     = 25
    BATCH_SIZE = 32
 
    start_time = time.time()
    model.train()
 
    for epoch in range(EPOCHS):
        epoch_loss = epoch_pred_loss = epoch_semantic_loss = 0.0
        batches    = 0
 
        permutation = torch.randperm(R_tensor.size(0))
        for i in range(0, R_tensor.size(0), BATCH_SIZE):
            indices    = permutation[i : i + BATCH_SIZE]
            batch_users = R_tensor[indices]
            batch_mask  = global_evaluation_mask[indices]
 
            t_loss, p_loss, s_loss = train_step(
                model, optimizer, batch_users, batch_mask,
                lambda_reg=0.05, epsilon_clip=0.15,
            )
 
            epoch_loss         += t_loss
            epoch_pred_loss    += p_loss
            epoch_semantic_loss += s_loss
            batches            += 1
 
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Total Loss: {epoch_loss / batches:.4f} | "
            f"Pred Loss (MSE): {epoch_pred_loss / batches:.4f} | "
            f"Semantic Loss (Drift): {epoch_semantic_loss / batches:.4f}"
        )
 
    print(f"-> Training done in {time.time() - start_time:.2f} seconds.")
 
    print("\nStep 6: Evaluating on a single user...")
    model.eval()
    test_user_idx      = 1
    single_user_vector = R_tensor[test_user_idx].unsqueeze(0)
 
    with torch.no_grad():
        ki_predictions, latent_profile = model(single_user_vector)
 
    print(f"-> Loaded user index {test_user_idx}.")
    print("\nTop 5 inferred genre interests:")
    profile_np        = latent_profile[0].numpy()
    sorted_genres_idx = np.argsort(profile_np)[::-1]
    for idx in sorted_genres_idx[:5]:
        print(f"   * {GENRES[idx]}: {profile_np[idx] * 100:.1f}%")
 
    print("\nStep 7: Simulating user override (HCAI intervention)...")
    user_ui_overrides = build_override_tensor(
        {"Sci-Fi": "strongly_increase", "Comedy": "strongly_reduce"},
        num_genres,
    )
 
    with torch.no_grad():
        hybrid_predictions, _ = model(single_user_vector, user_overrides=user_ui_overrides, alpha=3.0)
 
    print("\nOverride impact on selected films:")
 
    matrix_movie_id  = movies_df[movies_df["title"].str.contains("Matrix, The", na=False)]["movieId"].values[0]
    matrix_movie_idx = movie_id_to_idx[matrix_movie_id]
 
    hangover_movie_id  = movies_df[movies_df["title"].str.contains("Hangover, The", na=False)]["movieId"].values[0]
    hangover_movie_idx = movie_id_to_idx[hangover_movie_id]
 
    print(f"{idx_to_movie_title[matrix_movie_idx]} (Sci-Fi):")
    print(f"   Score without override: {ki_predictions[0, matrix_movie_idx]:.4f}")
    print(f"   Score with Sci-Fi (+2) override: {hybrid_predictions[0, matrix_movie_idx]:.4f}")
 
    print(f"{idx_to_movie_title[hangover_movie_idx]} (Comedy):")
    print(f"   Score without override: {ki_predictions[0, hangover_movie_idx]:.4f}")
    print(f"   Score with Comedy (-2) override: {hybrid_predictions[0, hangover_movie_idx]:.4f}")
 
    print("\nStep 8: Running XAI explanation engine...")
    xai_explanation = generate_soft_xai_explanation(
        movie_title=idx_to_movie_title[matrix_movie_idx],
        movie_idx=matrix_movie_idx,
        model=model,
        single_user_profile=latent_profile[0],
    )
    print(f"Explanation for user:\n-> \"{xai_explanation}\"")
 
    print("\nStep 9: Testing direct profile editing (HCAI Level 1)...")
    edited_profile = latent_profile.clone()
    edited_profile[0, GENRES.index("Sci-Fi")]   = 0.95
    edited_profile[0, GENRES.index("Comedy")]   = 0.05
    edited_profile[0, GENRES.index("Thriller")] = 0.80
 
    with torch.no_grad():
        profile_driven_output = model.recommend_from_edited_profile(edited_profile)
 
    print("-> User set: Sci-Fi=95%, Comedy=5%, Thriller=80%")
    print("-> Top 5 recommendations from manually edited profile:")
    top5 = torch.topk(profile_driven_output.squeeze(), 5)
    for rank, (score, idx) in enumerate(zip(top5.values, top5.indices), 1):
        title = idx_to_movie_title.get(idx.item(), f"Film #{idx.item()}")
        print(f"   {rank}. {title} -- Score: {score.item():.4f}")
 
    print("\nStep 10: Override impact analysis (HCAI Level 2)...")
    with torch.no_grad():
        impact = model.explain_override_impact(
            ann_output=ki_predictions,
            final_output=hybrid_predictions,
            idx_to_title=idx_to_movie_title,
            top_n=5,
        )
 
    print("-> Films most boosted by Sci-Fi override:")
    for title, delta in impact["boosted"]:
        print(f"   + {title}: +{delta:.4f}")
 
    print("-> Films most suppressed by Comedy override:")
    for title, delta in impact["suppressed"]:
        print(f"   - {title}: -{delta:.4f}")
 
    print("\nStep 11: Computing latent profiles for all users (XAI prep)...")
    model.eval()
    with torch.no_grad():
        _, all_latent_profiles = model(R_tensor)
    print(f"-> {all_latent_profiles.shape[0]} profiles computed (shape: {list(all_latent_profiles.shape)}).")
 
    print(f"\nStep 12: SHAP analysis for '{idx_to_movie_title[matrix_movie_idx]}'...")
    print("   (KernelSHAP on decoder - may take 10-30 seconds)")
    bg_indices      = torch.randperm(all_latent_profiles.size(0))[:30]
    background_prfs = all_latent_profiles[bg_indices]
 
    shap_result = explain_with_shap(
        model=model,
        user_profile=latent_profile[0],
        movie_idx=matrix_movie_idx,
        background_profiles=background_prfs,
        n_samples=100,
    )
    if shap_result:
        print("   SHAP values (genre contribution in stars):")
        for genre, val in shap_result[:6]:
            direction = "^" if val > 0 else "v"
            print(f"   [{direction}] {genre:<15} {val:+.4f} stars")
 
    print(f"\nStep 13: LIME analysis for '{idx_to_movie_title[matrix_movie_idx]}'...")
    lime_result = explain_with_lime(
        model=model,
        user_profile=latent_profile[0],
        movie_idx=matrix_movie_idx,
        n_samples=300,
    )
    if lime_result:
        print("   LIME coefficients (local genre importance, seed=42):")
        for genre, coef in lime_result[:6]:
            direction = "^" if coef > 0 else "v"
            print(f"   [{direction}] {genre:<15} {coef:+.4f}")
        print("   Note: LIME results may vary across runs due to random sampling.")
 
    print("\nStep 14: Global permutation feature importance (all users)...")
    perm_importance = compute_genre_permutation_importance(
        model=model,
        all_profiles=all_latent_profiles,
        all_targets=R_tensor,
        n_repeats=5,
    )
    print("   Genre importance (MSE increase when permuted):")
    for rank, (genre, delta) in enumerate(perm_importance.items(), 1):
        bar = "#" * max(1, int(delta * 200))
        print(f"   {rank:>2}. {genre:<15} delta MSE: {delta:+.5f}  {bar}")
 
    top_genre = next(iter(perm_importance))
    print(f"\n   -> Most important genre overall: '{top_genre}'")
 
    print("=" * 70)
 
# endregion
