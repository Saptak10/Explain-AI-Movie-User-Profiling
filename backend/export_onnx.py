"""
export_onnx.py
--------------
One-off conversion: takes the PyTorch checkpoint (vectorstore/model.pt,
trained via train.py) and exports it to ONNX for the `deployment` branch's
onnxruntime-based ai_service. Run this on `main` (where torch + the
checkpoint both live), not on `deployment` (which no longer installs torch).

Produces four files in vectorstore/onnx/:
  - model_standard.onnx     forward_standard:    (1, num_movies) -> predictions, latent_profile
  - model_interactive.onnx  forward_interactive: (1, num_genres) -> predictions
  - id_mapping.json         movie_id_to_idx / idx_to_movie_id / idx_to_title /
                            genres / genre_to_idx / num_movies / num_genres /
                            genre_mask -- everything ai_service needs that
                            isn't part of the ONNX graph itself.
  - importance_aux.npz      global_importance (num_genres,) and
                            movie_salience (num_movies,) -- precomputed here
                            from the raw weight matrices (encoder_l2 @
                            encoder_l1, decoder_l2 @ decoder_l1) since the
                            deployment branch's onnxruntime service has no
                            convenient way to re-derive them from the ONNX
                            graph's opaque initializers at runtime.

create_personalized_profile() (one-time fine-tune on a user's own ratings)
is NOT exported -- it does real gradient-descent training, which is
fundamentally incompatible with an inference-only ONNX runtime. The
deployment branch's /api/profile/personalize returns 501; the frontend
already falls back to the standard profile gracefully when that call fails
(see RatingsPage.jsx's handleSave).
"""

import json
from pathlib import Path

import numpy as np
import torch

from app.ai.id_mapping import IdMapping
from app.ai.model import DualModeHCAIAutoEncoder

CHECKPOINT_PATH = "vectorstore/model.pt"
OUT_DIR = Path("vectorstore/onnx")


def main():
    ck = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

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

    with torch.device("meta"):
        model = DualModeHCAIAutoEncoder(id_mapping, hidden_dim=ck["hidden_dim"])
    model.load_state_dict(ck["state_dict"], assign=True)
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exporting forward_standard ({id_mapping.num_movies} movies)...")
    example_x = torch.zeros(1, id_mapping.num_movies, dtype=torch.float32)

    class StandardWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            predictions, latent_profile = self.m.forward_standard(x)
            return predictions, latent_profile

    torch.onnx.export(
        StandardWrapper(model),
        (example_x,),
        str(OUT_DIR / "model_standard.onnx"),
        input_names=["x"],
        output_names=["predictions", "latent_profile"],
        dynamic_axes={"x": {0: "batch"}, "predictions": {0: "batch"}, "latent_profile": {0: "batch"}},
        opset_version=17,
    )

    print(f"Exporting forward_interactive ({id_mapping.num_genres} genres)...")
    example_override = torch.zeros(1, id_mapping.num_genres, dtype=torch.float32)

    class InteractiveWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, override_profile):
            return self.m.forward_interactive(override_profile)

    torch.onnx.export(
        InteractiveWrapper(model),
        (example_override,),
        str(OUT_DIR / "model_interactive.onnx"),
        input_names=["override_profile"],
        output_names=["predictions"],
        dynamic_axes={"override_profile": {0: "batch"}, "predictions": {0: "batch"}},
        opset_version=17,
    )

    print("Computing importance_aux.npz (global_importance, movie_salience)...")
    with torch.no_grad():
        eff = model.encoder_l2.weight @ model.encoder_l1.weight
        global_importance = eff.abs().mean(dim=1).numpy()

        decoder_eff = model.decoder_l2.weight @ model.decoder_l1.weight
        movie_salience = decoder_eff.abs().sum(dim=1).numpy()
    np.savez(
        OUT_DIR / "importance_aux.npz",
        global_importance=global_importance,
        movie_salience=movie_salience,
    )

    print("Writing id_mapping.json...")
    with open(OUT_DIR / "id_mapping.json", "w") as f:
        json.dump(
            {
                "movie_id_to_idx": id_mapping.movie_id_to_idx,
                "idx_to_movie_id": id_mapping.idx_to_movie_id,
                "idx_to_title": id_mapping.idx_to_title,
                "genres": id_mapping.genres,
                "genre_to_idx": id_mapping.genre_to_idx,
                "num_movies": id_mapping.num_movies,
                "num_genres": id_mapping.num_genres,
                "genre_mask": id_mapping.genre_mask,
            },
            f,
        )

    print("Done. Files in", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
