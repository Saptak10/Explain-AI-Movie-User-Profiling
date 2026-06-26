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

The encoder -> bottleneck -> decoder architecture, the weight-clipping
knowledge-injection strategy, and the override-impact / edited-profile
HCAI mechanics are carried over from the original SoftRegularizedHCAIAutoEncoder
design largely unchanged -- this refactor is about plumbing (dynamic sizes,
explicit mode routing, sparse-input hydration), not about redesigning the
HCAI architecture itself.
"""

from __future__ import annotations

import math

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
        genre_mask_tensor = torch.tensor(id_mapping.genre_mask, dtype=torch.float32)
        target_genre_matrix = genre_mask_tensor.T.contiguous()  # (num_genres, num_movies)
        self.register_buffer("target_genre_matrix", target_genre_matrix)

        # Encoder: Input Layer (movies) -> hidden -> Bottleneck (genres)
        self.encoder_l1 = nn.Linear(self.num_movies, hidden_dim, bias=False)
        self.encoder_act = nn.ReLU()
        self.encoder_l2 = nn.Linear(hidden_dim, self.num_genres, bias=False)

        # Decoder: Bottleneck (genres) -> hidden -> Output Layer (movies)
        self.dropout = nn.Dropout(p=0.4)
        self.decoder_l1 = nn.Linear(self.num_genres, hidden_dim)
        self.decoder_act = nn.ReLU()
        self.decoder_l2 = nn.Linear(hidden_dim, self.num_movies)

        self._inject_prior_knowledge(target_genre_matrix)

    def _inject_prior_knowledge(self, target_genre_matrix: torch.Tensor) -> None:
        """
        Hardcoded prior-knowledge weight injection: seeds the first
        num_genres rows of encoder_l1's weight with the real genre mask, and
        the first num_genres columns of encoder_l2's weight with an
        identity matrix, so the bottleneck starts out strictly bound to
        real-world genres before any gradient step has been taken.

        Guard: if hidden_dim < num_genres this injection cannot fit
        (there would not be enough hidden-layer rows/encoder_l2 columns to
        seed), so we raise early with a clear message rather than silently
        truncating the injected prior, which would corrupt the very
        knowledge-injection guarantee this architecture is built on.
        """
        if self.hidden_dim < self.num_genres:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be >= num_genres "
                f"({self.num_genres}) for the prior-knowledge injection to "
                f"fit into the hidden layer."
            )
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.encoder_l1.weight, a=math.sqrt(5))
            self.encoder_l1.weight[: self.num_genres, :].copy_(target_genre_matrix)
            nn.init.zeros_(self.encoder_l2.weight)
            self.encoder_l2.weight[:, : self.num_genres].copy_(
                torch.eye(self.num_genres)
            )

    # ------------------------------------------------------------------
    # Dual-Mode Forward
    # ------------------------------------------------------------------

    def forward_standard(self, x: torch.Tensor) -> tuple:
        """
        Standard Mode: complete end-to-end traversal.
            Input Layer (movies) -> Bottleneck (genres) -> Output Layer (movies)

        Args:
            x: (batch, num_movies) float32 -- sparse rating vectors
               (0.0 for unrated movies).

        Returns:
            (predicted_ratings, latent_genre_profile):
                predicted_ratings:    (batch, num_movies) in [0, 5]
                latent_genre_profile: (batch, num_genres) in [0, 1]
        """
        h_enc = self.encoder_act(self.encoder_l1(x))
        latent_profile = torch.sigmoid(self.encoder_l2(h_enc))

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
    # Knowledge-injection maintenance
    # ------------------------------------------------------------------

    def apply_weight_clipping(self, epsilon: float = 0.15) -> None:
        """
        Elastic-leash regularization: after each optimizer step, clamps the
        prior-knowledge-seeded weights back to within `epsilon` of their
        original semantic values, keeping the bottleneck nodes strictly
        bound to real-world genres throughout training rather than letting
        gradient descent freely drift them away from genre semantics.
        """
        with torch.no_grad():
            dev_l1 = (
                self.encoder_l1.weight[: self.num_genres, :] - self.target_genre_matrix
            )
            self.encoder_l1.weight[: self.num_genres, :].copy_(
                self.target_genre_matrix + torch.clamp(dev_l1, -epsilon, epsilon)
            )
            identity = torch.eye(self.num_genres, device=self.encoder_l2.weight.device)
            dev_l2 = self.encoder_l2.weight[:, : self.num_genres] - identity
            self.encoder_l2.weight[:, : self.num_genres].copy_(
                identity + torch.clamp(dev_l2, -epsilon, epsilon)
            )

    def get_semantic_loss(self) -> torch.Tensor:
        """
        Squared drift of the prior-knowledge-seeded weights from their
        semantic anchors. Uses torch.mean (not torch.sum): at ml-latest
        scale, num_movies ~ 87,000, so a sum over (num_genres * num_movies)
        elements would numerically dominate the prediction loss and stall
        learning -- exactly the bug the original implementation's
        docstring already called out, and which only gets worse as
        num_movies grows from ~9.7k to ~87k.
        """
        loss_l1 = torch.mean(
            (self.encoder_l1.weight[: self.num_genres, :] - self.target_genre_matrix) ** 2
        )
        identity = torch.eye(self.num_genres, device=self.encoder_l2.weight.device)
        loss_l2 = torch.mean(
            (self.encoder_l2.weight[:, : self.num_genres] - identity) ** 2
        )
        return loss_l1 + loss_l2

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
