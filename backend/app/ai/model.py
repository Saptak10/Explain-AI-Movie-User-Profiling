"""
model.py
--------
The Soft-Regularized HCAI Autoencoder, refactored from the ml-latest-small
prototype to:
  (a) take num_movies / num_genres / the genre mask dynamically from an
      IdMapping instead of a hardcoded 18-genre constant list, and
  (b) expose an explicit, named dual-mode forward pass instead of an
      implicit override-tensor branch, so the API layer can call either
      mode unambiguously.

Genre-bottleneck calibration (current design): earlier versions built the
genre logit by seeding encoder_l1/encoder_l2 with a hard 0/1 genre-mask
prior and elastically clipping them back toward it after every optimizer
step ("weight clipping"). That made the logit an unnormalized SUM of a
user's ratings in a genre -- a user with 1 rating and a user with 200
ratings in the same genre produced wildly different-scale logits that no
single temperature scalar could reconcile, and ~85% of the hidden layer
fed the bottleneck completely unregularized on top of that. The bottleneck
is now built from two additive parts instead (see forward_standard):
  1. An *anchored* signal computed directly (not learned) from
     target_genre_matrix -- a count-normalized, Bayesian-smoothed average
     rating per genre. This is exact and count-invariant by construction,
     so it needs no weight-clipping at all (there are no weights in it).
  2. A small, tanh-bounded *residual* signal from a freely-trained hidden
     layer, adding nuance without being able to dominate the anchored
     signal the way the old unregularized free pathway could.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from app.ai.id_mapping import IdMapping


class DualModeHCAIAutoEncoder(nn.Module):
    """
    Args:
        id_mapping: a fully-built IdMapping (defines num_movies, num_genres,
                    and the prior-knowledge genre_mask used for weight
                    injection/clipping).
        hidden_dim: width of the single hidden layer on each side of the
                    bottleneck.
    """

    def __init__(self, id_mapping: IdMapping, hidden_dim: int = 128):
        super().__init__()
        self.num_movies = id_mapping.num_movies
        self.num_genres = id_mapping.num_genres
        self.hidden_dim = hidden_dim

        if self.num_genres == 0:
            raise ValueError(
                "IdMapping discovered zero genres -- cannot build a "
                "genre-bottleneck autoencoder with a zero-width bottleneck. "
                "Check that movies.csv contains genre data."
            )

        # Prior-knowledge matrix: (num_genres, num_movies). Row g, column m
        # is 1.0 if movie m belongs to genre g. Built once here from the
        # IdMapping's (num_movies, num_genres) genre_mask via transpose.
        # Used directly in forward_standard's masked-mean pooling -- this
        # buffer IS the anchored genre signal's source of truth now, not
        # just an init/clip target for a learned weight matrix.
        genre_mask_tensor = torch.tensor(id_mapping.genre_mask, dtype=torch.float32)
        target_genre_matrix = genre_mask_tensor.T.contiguous()  # (num_genres, num_movies)
        self.register_buffer("target_genre_matrix", target_genre_matrix)

        # Residual encoder pathway only -- the anchored genre signal is a
        # closed-form computation in forward_standard (see module
        # docstring), not learned, so this no longer needs prior-knowledge
        # weight injection. Standard init; bounded via tanh at use-time.
        self.encoder_l1 = nn.Linear(self.num_movies, hidden_dim, bias=False)
        self.encoder_act = nn.ReLU()
        self.encoder_l2 = nn.Linear(hidden_dim, self.num_genres, bias=False)

        # Decoder: Bottleneck (genres) -> hidden -> Output Layer (movies)
        self.dropout = nn.Dropout(p=0.4)
        self.decoder_l1 = nn.Linear(self.num_genres, hidden_dim)
        self.decoder_act = nn.ReLU()
        self.decoder_l2 = nn.Linear(hidden_dim, self.num_movies)

        # Fixed (not learned) parameters of the closed-form anchored-signal
        # computation: count_smoothing is the Bayesian/Laplace pseudo-count
        # pulling low-rating-count genres toward rating_midpoint (the
        # neutral point of the 0-5 rating scale) instead of letting a
        # single rating swing the mean to its extreme.
        self.count_smoothing = 1.5
        self.rating_midpoint = 2.5

        # Learnable temperature dividing the anchored logit before the
        # sigmoid in forward_standard -- now scaling a count-normalized
        # average (bounded ~0-5 range) rather than an unbounded sum, so a
        # single global value can meaningfully calibrate it regardless of
        # how many ratings a user has in a given genre.
        self.genre_temperature = nn.Parameter(torch.tensor(5.0))

        # Learnable scale bounding the residual pathway's contribution to
        # the genre logit via tanh -- starts modest so the count-normalized
        # anchored signal dominates by default; clamped at use-time to
        # [0, 2.0] so it can grow with training but can never swamp the
        # anchored signal the way the old fully-unregularized free pathway
        # could (see get_semantic_loss for the complementary L2 penalty).
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    # ------------------------------------------------------------------
    # Dual-Mode Forward
    # ------------------------------------------------------------------

    def forward_standard(self, x: torch.Tensor) -> tuple:
        """
        Standard Mode: complete end-to-end traversal.
            Input Layer (movies) -> Bottleneck (genres) -> Output Layer (movies)

        The genre bottleneck logit is the sum of two parts (see module
        docstring for the motivation):
          1. Anchored: a Bayesian-smoothed, count-normalized average of the
             user's ratings within each genre, computed directly from
             target_genre_matrix -- exact and count-invariant, not learned.
          2. Residual: a small tanh-bounded signal from a freely-trained
             hidden layer, adding nuance without dominating the anchored
             signal.

        Args:
            x: (batch, num_movies) float32 -- sparse rating vectors
               (0.0 for unrated movies).

        Returns:
            (predicted_ratings, latent_genre_profile):
                predicted_ratings:    (batch, num_movies) in [0, 5]
                latent_genre_profile: (batch, num_genres) in [0, 1]
        """
        rated_mask = (x != 0).float()                              # (batch, num_movies)
        genre_sum = x @ self.target_genre_matrix.T                  # (batch, num_genres)
        genre_count = rated_mask @ self.target_genre_matrix.T       # (batch, num_genres)
        # Bayesian/Laplace-smoothed mean: shrinks toward rating_midpoint
        # when genre_count is 0 or small (count_smoothing acts as a prior
        # pseudo-count at the neutral rating), converges to the true
        # per-genre average as genre_count grows -- e.g. genre_count=0
        # gives exactly rating_midpoint (fully neutral), regardless of
        # temperature.
        genre_mean = (
            genre_sum + self.count_smoothing * self.rating_midpoint
        ) / (genre_count + self.count_smoothing)

        temperature = self.genre_temperature.clamp(min=0.5)
        anchored_logit = (genre_mean - self.rating_midpoint) / temperature

        h_enc = self.encoder_act(self.encoder_l1(x))
        residual_scale = self.residual_scale.clamp(min=0.0, max=2.0)
        residual_logit = torch.tanh(self.encoder_l2(h_enc)) * residual_scale

        latent_profile = torch.sigmoid(anchored_logit + residual_logit)

        h_dec = self.decoder_act(self.decoder_l1(self.dropout(latent_profile)))
        predicted_ratings = torch.sigmoid(self.decoder_l2(h_dec)) * 5.0
        return predicted_ratings, latent_profile

    def forward_interactive(self, override_profile: torch.Tensor) -> torch.Tensor:
        """
        Interactive Profile Mode: completely bypasses the encoder (Input
        Layer -> Bottleneck half of the network). Accepts a genre-space
        vector directly from the API -- representing a user manually
        dragging genre sliders in the UI -- injects it straight into the
        bottleneck, and passes it through the decoder to produce updated
        recommendations in real time.

        Args:
            override_profile: (batch, num_genres) float32, values typically
                               in [0, 1] (one value per discovered genre).

        Returns:
            predicted_ratings: (batch, num_movies) in [0, 5].

        Raises:
            ValueError: if override_profile's last dimension does not match
                        num_genres, since silently broadcasting/truncating
                        a mismatched genre vector would inject the wrong
                        genre's slider value into the wrong bottleneck node.
        """
        if override_profile.shape[-1] != self.num_genres:
            raise ValueError(
                f"override_profile last dim ({override_profile.shape[-1]}) "
                f"must equal num_genres ({self.num_genres})."
            )
        h_dec = self.decoder_act(self.decoder_l1(override_profile))
        predicted_ratings = torch.sigmoid(self.decoder_l2(h_dec)) * 5.0
        return predicted_ratings

    def forward(self, x: torch.Tensor = None, override_profile: torch.Tensor = None):
        """
        Convenience dispatcher so existing call sites that expect a single
        `model(x)`-style call keep working. Exactly one of `x` or
        `override_profile` must be supplied -- supplying both or neither is
        a caller error, since it would leave the execution mode ambiguous.

        Prefer calling forward_standard / forward_interactive directly in
        new code; this dispatcher exists for drop-in compatibility.
        """
        if x is not None and override_profile is not None:
            raise ValueError(
                "Provide exactly one of `x` (Standard Mode) or "
                "`override_profile` (Interactive Mode), not both."
            )
        if x is not None:
            return self.forward_standard(x)
        if override_profile is not None:
            return self.forward_interactive(override_profile)
        raise ValueError("Provide either `x` or `override_profile`.")

    # ------------------------------------------------------------------
    # Residual-pathway regularization
    # ------------------------------------------------------------------
    #
    # No apply_weight_clipping() anymore: the anchored genre signal is a
    # closed-form computation from target_genre_matrix (see
    # forward_standard), not a learned weight matrix, so there is nothing
    # left to clip toward a prior -- it cannot drift by construction.

    def get_semantic_loss(self) -> torch.Tensor:
        """
        L2 penalty on the residual encoder pathway's weights (encoder_l1,
        encoder_l2). These no longer anchor to a hard genre prior -- that
        anchoring is now exact and weight-free (see forward_standard) -- so
        this term instead discourages the *residual* pathway from growing
        large, complementing the tanh/residual_scale bound in
        forward_standard so the residual can add nuance without dominating
        the count-normalized anchored signal. Uses torch.mean (not
        torch.sum): at ml-latest scale, num_movies ~ 87,000, so a sum would
        numerically dominate the prediction loss and stall learning.
        """
        return (
            torch.mean(self.encoder_l1.weight ** 2)
            + torch.mean(self.encoder_l2.weight ** 2)
        )

    # ------------------------------------------------------------------
    # HCAI explainability helpers (operate on a forward_standard() result)
    # ------------------------------------------------------------------

    def extract_taste_profile(
        self, latent_profile: torch.Tensor, genres: list
    ) -> dict:
        """
        Human-Centered XAI Profile Extraction: takes the (batch, num_genres)
        latent activations from forward_standard() and turns them into a
        UI-ready "User Taste Profile" -- a plain dict of
        {genre_name: percentage_match}, sorted descending.

        Only supports a single user (batch size 1) per call, since a "User
        Taste Profile" is, by definition, one user's profile; batched
        callers should call this once per row.

        Args:
            latent_profile: (1, num_genres) or (num_genres,) tensor of
                             sigmoid-activated bottleneck values in [0, 1].
            genres: ordered genre name list (must match IdMapping.genres,
                    i.e. column order of the bottleneck).

        Returns:
            Dict {genre_name: float percentage in [0, 100]}, ordered by
            descending match strength.
        """
        flat = latent_profile.detach().reshape(-1)
        if flat.shape[0] != len(genres):
            raise ValueError(
                f"latent_profile has {flat.shape[0]} entries but {len(genres)} "
                f"genre names were provided."
            )
        pairs = [(genres[i], float(flat[i].item()) * 100.0) for i in range(len(genres))]
        pairs.sort(key=lambda gp: gp[1], reverse=True)
        return dict(pairs)

    def explain_override_impact(
        self,
        ann_output: torch.Tensor,
        final_output: torch.Tensor,
        idx_to_title: dict,
        top_n: int = 5,
    ) -> dict:
        """
        Delta-transparency for an Interactive Mode override: compares the
        Standard-Mode output against an Interactive-Mode (or hybrid)
        output and surfaces the top_n most boosted and most suppressed
        movies by score delta, with titles resolved via idx_to_title.
        """
        delta = (final_output - ann_output).reshape(-1)
        k = min(top_n, delta.shape[0])

        top_up_vals, top_up_idx = torch.topk(delta, k)
        top_down_vals, top_down_idx = torch.topk(-delta, k)

        boosted = [
            (idx_to_title.get(i.item(), f"Movie #{i.item()}"), v.item())
            for i, v in zip(top_up_idx, top_up_vals)
        ]
        suppressed = [
            (idx_to_title.get(i.item(), f"Movie #{i.item()}"), -v.item())
            for i, v in zip(top_down_idx, top_down_vals)
        ]
        return {"boosted": boosted, "suppressed": suppressed}
