"""
train.py
--------
Standalone one-time training script for DualModeHCAIAutoEncoder.

Run from the backend/ directory:
    python train.py
    python train.py --epochs 30 --batch-size 64 --lr 0.005

What this script does:
  1. Reads movies.csv once to build the IdMapping (genre vocabulary + ID
     translation layer).
  2. Reads the ratings.npy binary dataset fully into memory for splitting 
     (this is highly memory efficient and avoids all string parsing).
  3. Shuffles data with a fixed seed, then writes three temporary .npy files:
     train (80%), val (10%), test (10%).
  4. Trains the model using the streaming SparseUserVectorDataset pipeline
     (no dense ratings matrix — peak RAM during training is O(batch_size
     * num_movies) per the existing pipeline design).
  5. Evaluates regression metrics (RMSE, MAE, R²) on the validation set
     after every epoch.
  6. Evaluates all 7 metrics on the test set once at the end:
       - Regression: RMSE, MAE, R²
       - Ranking (Precision@N, Recall@N, F1@N, Accuracy@N): treats
         ratings >= LIKE_THRESHOLD as "liked"; checks whether the top-N
         predicted items the user had not seen in training overlap with
         the liked items in the test set.
  7. Saves an interactive Plotly HTML dashboard to --output-dir.
  8. Saves a checkpoint compatible with ai_service.load():
       {state_dict, movie_id_to_idx, idx_to_movie_id, idx_to_title,
        genres, genre_to_idx, num_movies, num_genres, genre_mask, hidden_dim}
     PLUS optimizer_state_dict and train_history for fine-tuning resumption.

Fine-tuning after initial training:
    python train.py --resume path/to/model.pt --epochs 5

The --resume flag loads the full checkpoint (model weights + optimizer
state), re-uses the IdMapping embedded in it (so the same dense-index
space is preserved), and continues training on the same ratings dataset.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ── Project imports (run from backend/ with PYTHONPATH=. or python train.py)
sys.path.insert(0, str(Path(__file__).parent))
from app.ai.id_mapping import IdMapping, build_id_mapping
from app.ai.model import DualModeHCAIAutoEncoder
from app.ai.losses import train_step
from app.ai.data_pipeline import SparseUserVectorDataset, hydrate_user_vector

# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants
# ─────────────────────────────────────────────────────────────────────────────

# Rating threshold above which a movie is considered "liked" for the
# classification-style metrics (Accuracy, Precision, Recall, F1).
# Change this in one place; it propagates to all metric computations.
LIKE_THRESHOLD: float = 4.0

# How many top predictions to evaluate ranking metrics at.
TOP_N: int = 10

# Train / val / test fractions (must sum to 1.0).
TRAIN_FRAC: float = 0.80
VAL_FRAC:   float = 0.10
TEST_FRAC:  float = 0.10

# Default paths (can be overridden via CLI arguments).
DEFAULT_MOVIES_CSV:  str = "app/database/ml-latest/movies.csv"
DEFAULT_RATINGS_NPY: str = "app/database/ml-latest/ratings.npy"
DEFAULT_OUTPUT_DIR:  str = "vectorstore"
DEFAULT_MODEL_NAME:  str = "model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DualModeHCAIAutoEncoder")
    p.add_argument("--movies-csv",  default=DEFAULT_MOVIES_CSV)
    p.add_argument("--ratings-npy", default=DEFAULT_RATINGS_NPY)
    p.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-name",  default=DEFAULT_MODEL_NAME)
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=512)
    p.add_argument("--lr",          type=float, default=0.01)
    p.add_argument("--hidden-dim",  type=int,   default=128)
    p.add_argument("--mask-fraction", type=float, default=0.2)
    p.add_argument("--lambda-reg",  type=float, default=0.05)
    p.add_argument("--epsilon-clip",type=float, default=0.15)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to an existing checkpoint to resume training from.")
    p.add_argument("--top-n",       type=int,   default=TOP_N,
                   help="N for Precision/Recall/F1/Accuracy @N metrics.")
    p.add_argument("--like-threshold", type=float, default=LIKE_THRESHOLD,
                   help="Minimum rating to count as 'liked' for ranking metrics.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data splitting (Binary/NumPy)
# ─────────────────────────────────────────────────────────────────────────────

def create_binary_splits(npy_path: str, seed: int):
    """
    Loads the full memory-mapped binary dataset, shuffles it, splits it 
    80/10/10, and sorts each split by userId to satisfy the streaming 
    UserAggregator's assumptions.
    
    Returns the paths to the temp .npy files, and the in-memory arrays 
    for the validation and test evaluation steps.
    """
    print(f"Loading {npy_path} into memory for splitting…")
    # Load the structured array into RAM
    data = np.load(npy_path)
    
    print("  Shuffling data…")
    rng = np.random.default_rng(seed)
    rng.shuffle(data)
    
    n = len(data)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    
    train_data = data[:n_train]
    val_data   = data[n_train : n_train + n_val]
    test_data  = data[n_train + n_val :]
    
    print(
        f"  Split: {len(train_data):,} train / "
        f"{len(val_data):,} val / "
        f"{len(test_data):,} test rows."
    )
    
    print("  Sorting splits by userId…")
    # NumPy structured arrays can be sorted directly by field name in C
    train_data.sort(order='userId')
    val_data.sort(order='userId')
    test_data.sort(order='userId')
    
    def write_temp_npy(arr, suffix):
        fd, path = tempfile.mkstemp(suffix=f"_{suffix}.npy", prefix="hcai_")
        os.close(fd)
        np.save(path, arr)
        return path

    print("  Writing temporary split NPY files…")
    train_npy = write_temp_npy(train_data, "train")
    val_npy   = write_temp_npy(val_data, "val")
    test_npy  = write_temp_npy(test_data, "test")
    
    return train_npy, val_npy, test_npy, train_data, val_data, test_data


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_regression_metrics(
    model: DualModeHCAIAutoEncoder,
    rows: np.ndarray,
    id_mapping: IdMapping,
) -> dict[str, float]:
    """
    Computes RMSE, MAE, and R² on a set of (userId, movieId, rating) rows.
    """
    model.eval()

    # Group rows by user
    user_ratings: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        uid = int(row["userId"])
        mid = int(row["movieId"])
        r   = float(row["rating"])
        dense_idx = id_mapping.movie_id_to_dense(mid)
        if dense_idx is None:
            continue
        user_ratings[uid].append((dense_idx, r))

    sum_sq_err   = 0.0
    sum_abs_err  = 0.0
    sum_target   = 0.0
    sum_target_sq = 0.0
    n_total      = 0

    with torch.no_grad():
        for uid, pairs in user_ratings.items():
            if not pairs:
                continue
            vec, _ = hydrate_user_vector(pairs, id_mapping.num_movies)
            x = vec.unsqueeze(0).to(device)  # (1, num_movies)
            predictions, _ = model.forward_standard(x)
            preds = predictions[0]  # (num_movies,)

            for dense_idx, true_rating in pairs:
                pred_rating = float(preds[dense_idx].item())
                err = pred_rating - true_rating
                sum_sq_err   += err * err
                sum_abs_err  += abs(err)
                sum_target   += true_rating
                sum_target_sq += true_rating * true_rating
                n_total      += 1

    if n_total == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    rmse = math.sqrt(sum_sq_err / n_total)
    mae  = sum_abs_err / n_total
    mean_target = sum_target / n_total
    ss_tot = sum_target_sq - n_total * mean_target * mean_target
    r2 = 1.0 - (sum_sq_err / ss_tot) if ss_tot > 1e-9 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def compute_ranking_metrics(
    model: DualModeHCAIAutoEncoder,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    id_mapping: IdMapping,
    top_n: int,
    like_threshold: float,
) -> dict[str, float]:
    """
    Computes Precision@N, Recall@N, F1@N, and Accuracy@N on the test set.
    """
    model.eval()

    # Build per-user train item sets (the input context)
    train_items: dict[int, list[tuple[int, float]]] = defaultdict(list)
    train_movie_ids_by_user: dict[int, set[int]] = defaultdict(set)
    for row in train_rows:
        uid = int(row["userId"])
        mid = int(row["movieId"])
        dense_idx = id_mapping.movie_id_to_dense(mid)
        if dense_idx is None:
            continue
        r = float(row["rating"])
        train_items[uid].append((dense_idx, r))
        train_movie_ids_by_user[uid].add(dense_idx)

    # Build per-user test liked sets
    test_liked: dict[int, set[int]] = defaultdict(set)
    for row in test_rows:
        uid = int(row["userId"])
        mid = int(row["movieId"])
        r   = float(row["rating"])
        dense_idx = id_mapping.movie_id_to_dense(mid)
        if dense_idx is None:
            continue
        if r >= like_threshold:
            test_liked[uid].add(dense_idx)

    precisions: list[float] = []
    recalls:    list[float] = []
    f1s:        list[float] = []
    hits:       list[float] = []

    with torch.no_grad():
        for uid, liked_set in test_liked.items():
            if not liked_set:
                continue

            # Build the user's input from train rows
            train_pairs = train_items.get(uid, [])
            if train_pairs:
                vec, _ = hydrate_user_vector(train_pairs, id_mapping.num_movies)
            else:
                vec = torch.zeros(id_mapping.num_movies, dtype=torch.float32)

            x = vec.unsqueeze(0).to(device)
            predictions, _ = model.forward_standard(x)
            scores = predictions[0]

            # Mask out training items — model should not recommend
            # things the user already rated in training
            train_set = train_movie_ids_by_user.get(uid, set())
            for idx in train_set:
                scores[idx] = -1.0  # push below any real prediction

            top_indices = set(
                torch.argsort(scores, descending=True)[:top_n].tolist()
            )

            n_relevant_in_top = len(top_indices & liked_set)
            precision = n_relevant_in_top / top_n
            recall    = n_relevant_in_top / len(liked_set)
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            hit = 1.0 if n_relevant_in_top > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            hits.append(hit)

    if not precisions:
        nan = float("nan")
        return {"precision": nan, "recall": nan, "f1": nan, "accuracy": nan}

    return {
        "precision": sum(precisions) / len(precisions),
        "recall":    sum(recalls)    / len(recalls),
        "f1":        sum(f1s)        / len(f1s),
        "accuracy":  sum(hits)       / len(hits),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotly dashboard
# ─────────────────────────────────────────────────────────────────────────────

def build_plotly_dashboard(
    history: dict,
    test_metrics: dict,
    output_path: str,
    top_n: int,
    like_threshold: float,
) -> None:
    """
    Builds and saves an interactive Plotly HTML dashboard containing:
      Row 1: Training loss + Validation loss (per epoch)
      Row 2: Validation RMSE + Validation MAE (per epoch)
      Row 3: Validation R² (per epoch)
      Row 4: Final test-set metrics bar chart (all 7 metrics)

    history keys expected:
        epochs, train_loss, val_rmse, val_mae, val_r2

    Requires plotly to be installed: pip install plotly
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print(
            "  [WARNING] plotly not installed — skipping HTML dashboard. "
            "Run: pip install plotly"
        )
        return

    epochs = history["epochs"]

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Training Loss (masked MSE + semantic drift)",
            "Validation RMSE",
            "Validation MAE",
            "Validation R²",
            "",
            "",
            f"Final Test-Set Metrics  |  @{top_n}  |  like ≥ {like_threshold}",
            "",
        ),
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
            [{}, {}],
            [{"colspan": 2}, None],
        ],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    # ── Row 1: training loss ────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history["train_loss"],
            mode="lines+markers", name="Train Loss",
            line=dict(color="#636EFA", width=2),
            marker=dict(size=5),
            hovertemplate="Epoch %{x}<br>Train Loss: %{y:.5f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── Row 2: RMSE + MAE ───────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history["val_rmse"],
            mode="lines+markers", name="Val RMSE",
            line=dict(color="#EF553B", width=2),
            marker=dict(size=5),
            hovertemplate="Epoch %{x}<br>Val RMSE: %{y:.5f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history["val_mae"],
            mode="lines+markers", name="Val MAE",
            line=dict(color="#00CC96", width=2),
            marker=dict(size=5),
            hovertemplate="Epoch %{x}<br>Val MAE: %{y:.5f}<extra></extra>",
        ),
        row=2, col=2,
    )

    # ── Row 3: R² ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=epochs, y=history["val_r2"],
            mode="lines+markers", name="Val R²",
            line=dict(color="#AB63FA", width=2),
            marker=dict(size=5),
            hovertemplate="Epoch %{x}<br>Val R²: %{y:.5f}<extra></extra>",
        ),
        row=3, col=1,
    )
    # Add R² = 0 reference line
    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", line_width=1, row=3, col=1
    )

    # ── Row 4: final test-set bar chart ─────────────────────────────────────
    metric_labels = [
        f"Accuracy@{top_n}", f"Precision@{top_n}",
        f"Recall@{top_n}", f"F1@{top_n}",
        "RMSE", "MAE", "R²",
    ]
    metric_values = [
        test_metrics.get("accuracy",  float("nan")),
        test_metrics.get("precision", float("nan")),
        test_metrics.get("recall",    float("nan")),
        test_metrics.get("f1",        float("nan")),
        test_metrics.get("rmse",      float("nan")),
        test_metrics.get("mae",       float("nan")),
        test_metrics.get("r2",        float("nan")),
    ]
    bar_colors = [
        "#636EFA", "#636EFA", "#636EFA", "#636EFA",  # ranking metrics: blue
        "#EF553B", "#00CC96", "#AB63FA",              # regression metrics
    ]
    fig.add_trace(
        go.Bar(
            x=metric_labels,
            y=metric_values,
            marker_color=bar_colors,
            text=[
                f"{v:.4f}" if not math.isnan(v) else "N/A"
                for v in metric_values
            ],
            textposition="outside",
            name="Test Metrics",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ),
        row=4, col=1,
    )

    fig.update_layout(
        title=dict(
            text="HCAI Autoencoder — Training Dashboard",
            font=dict(size=20),
            x=0.5,
        ),
        height=1100,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", y=-0.02),
    )

    # Axis labels
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss",  row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="RMSE",  row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    fig.update_yaxes(title_text="MAE",   row=2, col=2)
    fig.update_xaxes(title_text="Epoch", row=3, col=1)
    fig.update_yaxes(title_text="R²",    row=3, col=1)
    fig.update_yaxes(title_text="Score", row=4, col=1)

    fig.write_html(output_path)
    print(f"  Dashboard saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: str,
    model: DualModeHCAIAutoEncoder,
    optimizer: optim.Optimizer,
    id_mapping: IdMapping,
    train_history: dict,
    args: argparse.Namespace,
) -> None:
    """
    Saves a checkpoint that:
      - Is fully compatible with ai_service.load() (all expected keys present).
      - Also stores optimizer_state_dict and train_history so training can
        be resumed with --resume without losing the optimizer's momentum.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # ── keys required by ai_service.load() ──────────────────────
            "state_dict":       model.state_dict(),
            "movie_id_to_idx":  id_mapping.movie_id_to_idx,
            "idx_to_movie_id":  id_mapping.idx_to_movie_id,
            "idx_to_title":     id_mapping.idx_to_title,
            "genres":           id_mapping.genres,
            "genre_to_idx":     id_mapping.genre_to_idx,
            "num_movies":       id_mapping.num_movies,
            "num_genres":       id_mapping.num_genres,
            "genre_mask":       id_mapping.genre_mask,
            "hidden_dim":       model.hidden_dim,
            # ── extra keys for resumption ────────────────────────────────
            "optimizer_state_dict": optimizer.state_dict(),
            "train_history":    train_history,
            "args":             vars(args),
        },
        path,
    )
    print(f"  Checkpoint saved → {path}")


def load_checkpoint_for_resume(
    path: str,
    args: argparse.Namespace,
) -> tuple[DualModeHCAIAutoEncoder, optim.Optimizer, IdMapping, dict]:
    """
    Loads an existing checkpoint for fine-tuning resumption.
    Reconstructs the IdMapping from the stored dicts (not from movies.csv),
    so the dense index space is guaranteed identical to the original run.
    The learning rate and other hyperparameters come from the current CLI
    args, not the saved args, so you can change them on resume.
    """
    print(f"Resuming from checkpoint: {path}")
    ck = torch.load(path, weights_only=False)

    id_mapping = IdMapping(
        movie_id_to_idx = ck["movie_id_to_idx"],
        idx_to_movie_id = ck["idx_to_movie_id"],
        idx_to_title    = ck["idx_to_title"],
        genres          = ck["genres"],
        genre_to_idx    = ck["genre_to_idx"],
        num_movies      = ck["num_movies"],
        num_genres      = ck["num_genres"],
        genre_mask      = ck["genre_mask"],
    )

    model = DualModeHCAIAutoEncoder(id_mapping, hidden_dim=ck["hidden_dim"])
    model.load_state_dict(ck["state_dict"])
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    if "optimizer_state_dict" in ck:
        optimizer.load_state_dict(ck["optimizer_state_dict"])

    history = ck.get("train_history", {
        "epochs": [], "train_loss": [], "val_rmse": [], "val_mae": [], "val_r2": []
    })
    print(
        f"  Resumed: {id_mapping.num_movies} movies, "
        f"{id_mapping.num_genres} genres, "
        f"hidden_dim={ck['hidden_dim']}, "
        f"prior epochs={len(history['epochs'])}"
    )
    return model, optimizer, id_mapping, history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(output_dir / args.model_name)
    dashboard_path  = str(output_dir / "training_dashboard.html")

    # ── 1. Read and split binary dataset ───────────────────────────────────
    tmp_files: list[str] = []
    try:
        print("Writing temporary split NPY files…")
        train_npy, val_npy, test_npy, train_data, val_data, test_data = create_binary_splits(
            args.ratings_npy, seed=args.seed
        )
        tmp_files.extend([train_npy, val_npy, test_npy])
        print(f"  train temp file → {train_npy}")

        # ── 2. Build IdMapping from movies.csv (or reuse from checkpoint) ─
        if args.resume:
            model, optimizer, id_mapping, history = load_checkpoint_for_resume(
                args.resume, args
            )
        else:
            print(f"\nBuilding ID mapping from {args.movies_csv}…")
            id_mapping = build_id_mapping(args.movies_csv)
            print(
                f"  {id_mapping.num_movies:,} movies, "
                f"{id_mapping.num_genres} genres discovered."
            )
            model = DualModeHCAIAutoEncoder(id_mapping, hidden_dim=args.hidden_dim)
            model = model.to(device)
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=1e-5
            )
            history: dict = {
                "epochs":    [],
                "train_loss": [],
                "val_rmse":  [],
                "val_mae":   [],
                "val_r2":    [],
            }

        epoch_offset = len(history["epochs"])  # 0 on fresh run, >0 on resume

        # ── 3. Training loop ───────────────────────────────────────────────
        dataset = SparseUserVectorDataset(
            train_npy,
            id_mapping,
            mask_fraction=args.mask_fraction,
            seed=args.seed,
        )

        print(
            f"\nTraining for {args.epochs} epoch(s) "
            f"(batch_size={args.batch_size}, lr={args.lr}, "
            f"mask_fraction={args.mask_fraction})…"
        )
        best_val_rmse  = float("inf")
        best_ckpt_path = checkpoint_path.replace(".pt", "_best.pt")

        for epoch in range(1, args.epochs + 1):
            epoch_abs = epoch + epoch_offset  # absolute epoch number for history
            t0 = time.time()
            model.train()

            loader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                pin_memory=True, 
                num_workers=4, 
                prefetch_factor=2
            )
            running_loss = 0.0
            n_batches    = 0

            for batch in loader:
                total_loss, _, _ = train_step(
                    model,
                    optimizer,
                    batch["input_vector"].to(device),
                    batch["target_vector"].to(device),
                    batch["hidden_mask"].to(device),
                    lambda_reg=args.lambda_reg,
                    epsilon_clip=args.epsilon_clip,
                )
                running_loss += total_loss
                n_batches    += 1

            avg_train_loss = running_loss / max(n_batches, 1)

            # ── Validation metrics (every epoch) ──────────────────────────
            val_m = compute_regression_metrics(model, val_data, id_mapping)

            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch_abs:03d}/{epoch_abs - epoch + args.epochs}  "
                f"loss={avg_train_loss:.5f}  "
                f"val_rmse={val_m['rmse']:.4f}  "
                f"val_mae={val_m['mae']:.4f}  "
                f"val_r2={val_m['r2']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            history["epochs"].append(epoch_abs)
            history["train_loss"].append(avg_train_loss)
            history["val_rmse"].append(val_m["rmse"])
            history["val_mae"].append(val_m["mae"])
            history["val_r2"].append(val_m["r2"])

            # Save best checkpoint (by val RMSE)
            if val_m["rmse"] < best_val_rmse:
                best_val_rmse = val_m["rmse"]
                save_checkpoint(
                    best_ckpt_path, model, optimizer, id_mapping, history, args
                )
                print(f"    ↑ New best val RMSE={best_val_rmse:.4f} — saved best checkpoint.")

        model.eval()

        # ── 4. Final test-set evaluation ────────────────────────────────────
        print("\nEvaluating on test set…")
        test_regression = compute_regression_metrics(model, test_data, id_mapping)
        test_ranking    = compute_ranking_metrics(
            model, train_data, test_data, id_mapping,
            top_n=args.top_n,
            like_threshold=args.like_threshold,
        )
        test_metrics = {**test_regression, **test_ranking}

        print("\n── Test Set Results ─────────────────────────────────────────")
        print(f"  RMSE:            {test_metrics['rmse']:.4f}")
        print(f"  MAE:             {test_metrics['mae']:.4f}")
        print(f"  R²:              {test_metrics['r2']:.4f}")
        print(f"  Accuracy@{args.top_n}:    {test_metrics['accuracy']:.4f}  (hit rate)")
        print(f"  Precision@{args.top_n}:   {test_metrics['precision']:.4f}")
        print(f"  Recall@{args.top_n}:      {test_metrics['recall']:.4f}")
        print(f"  F1@{args.top_n}:          {test_metrics['f1']:.4f}")
        print(f"  (like_threshold=≥{args.like_threshold}, top_n={args.top_n})")
        print("────────────────────────────────────────────────────────────")

        # ── 5. Save final checkpoint (last epoch, compatible with ai_service)
        print("\nSaving final checkpoint…")
        save_checkpoint(checkpoint_path, model, optimizer, id_mapping, history, args)

        # ── 6. Plotly HTML dashboard ─────────────────────────────────────
        print("Building Plotly dashboard…")
        build_plotly_dashboard(
            history, test_metrics, dashboard_path,
            top_n=args.top_n,
            like_threshold=args.like_threshold,
        )

        print(f"\nDone. Files written to {output_dir}/")
        print(f"  Final checkpoint : {checkpoint_path}")
        print(f"  Best checkpoint  : {best_ckpt_path}")
        print(f"  Dashboard        : {dashboard_path}")
        print(
            "\nTo load in FastAPI, ensure settings.model_save_path points to "
            f"{checkpoint_path} and restart the server. ai_service.load() "
            "will pick it up automatically."
        )

    finally:
        # Always clean up temp .npy files, even if an exception occurred
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    main()