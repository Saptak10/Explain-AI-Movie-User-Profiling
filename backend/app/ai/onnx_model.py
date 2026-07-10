"""
onnx_model.py
-------------
Inference-only stand-in for DualModeHCAIAutoEncoder (model.py), backed by
onnxruntime instead of torch. Exists only on the `deployment` branch: this
branch never trains, it only serves a checkpoint that was already exported
to ONNX on `main` (see export_onnx.py). Dropping torch for a plain
onnxruntime + numpy stack is what actually fits a memory-constrained free
host -- `import torch` alone costs ~200MB RSS before touching any data,
which was the dominant cost that kept the full-PyTorch service over the
512MB free-tier ceiling even after every other optimization.

Exposes the same forward_standard/forward_interactive/extract_taste_profile
surface as the real model so ai_service.py and xai.py need only swap torch
tensor ops for numpy ones, not restructure their control flow.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort


class OnnxDualModeModel:
    def __init__(self, standard_path: str, interactive_path: str, num_movies: int, num_genres: int):
        self.num_movies = num_movies
        self.num_genres = num_genres
        # Single-threaded intra-op: this service handles one request's
        # forward pass at a time per worker anyway (see asyncio.to_thread
        # call sites in ai_service.py), so extra onnxruntime-internal
        # threads would only add memory/scheduling overhead for no
        # latency benefit on a 1-vCPU free-tier instance.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._standard_sess = ort.InferenceSession(
            standard_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._interactive_sess = ort.InferenceSession(
            interactive_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def forward_standard(self, x: np.ndarray) -> tuple:
        """x: (1, num_movies) float32 -> (predictions, latent_profile), both numpy."""
        predictions, latent_profile = self._standard_sess.run(
            None, {"x": x.astype(np.float32)}
        )
        return predictions, latent_profile

    def forward_interactive(self, override_profile: np.ndarray) -> np.ndarray:
        """override_profile: (1, num_genres) float32 -> predictions, numpy."""
        if override_profile.shape[-1] != self.num_genres:
            raise ValueError(
                f"override_profile last dim ({override_profile.shape[-1]}) "
                f"must equal num_genres ({self.num_genres})."
            )
        (predictions,) = self._interactive_sess.run(
            None, {"override_profile": override_profile.astype(np.float32)}
        )
        return predictions

    def extract_taste_profile(self, latent_profile: np.ndarray, genres: list) -> dict:
        """Same contract as DualModeHCAIAutoEncoder.extract_taste_profile."""
        flat = np.asarray(latent_profile).reshape(-1)
        if flat.shape[0] != len(genres):
            raise ValueError(
                f"latent_profile has {flat.shape[0]} entries but {len(genres)} "
                f"genre names were provided."
            )
        pairs = [(genres[i], float(flat[i]) * 100.0) for i in range(len(genres))]
        pairs.sort(key=lambda gp: gp[1], reverse=True)
        return dict(pairs)
