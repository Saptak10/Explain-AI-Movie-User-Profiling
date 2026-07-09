# Setup guide (fresh clone from GitHub)

The dataset and the trained model are **not** in the git repo (both are gitignored — the ratings
data alone is ~900MB and the model checkpoint is ~280MB, well past anything that belongs in git).
You get them from teammates / re-download them separately, as described below.

> The root `README.md` is stale (describes an old MongoDB/LangChain architecture this project no
> longer uses) — ignore it and follow this file instead.

## Prerequisites

- **Python 3.12** (the pinned CUDA torch wheel in `requirements.txt` is built specifically for
  Python 3.12 — `cp312`. If you're on a different Python version, see the GPU/CPU note below.)
- **Node.js 20+** (tested with Node 24)
- Optional: an NVIDIA GPU + recent driver, if you want training to run on GPU (recommended — see below)

## 1. Clone and install dependencies

```bash
git clone <repo-url>
cd Explain-AI-Movie-User-Profiling

# Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

**GPU vs CPU note:** `requirements.txt` pins `torch==2.12.0+cu130` with
`--extra-index-url https://download.pytorch.org/whl/cu130`, which requires an NVIDIA GPU with a
CUDA 13.x-compatible driver. If you don't have an NVIDIA GPU, edit `requirements.txt` before
installing: delete the `--extra-index-url` line and change `torch==2.12.0+cu130` to `torch==2.12.0`
(installs the CPU-only build from plain PyPI instead). The app runs fine on CPU — only training
speed is affected (GPU: ~8 min/epoch on the full dataset on an RTX 3060 Ti; CPU: expect several
times slower).

```bash
# Frontend, separate terminal
cd frontend
npm install
```

## 2. Get a trained model (the fast path — no dataset download needed)

If a teammate can share their trained checkpoint, this is by far the easiest way to get running:

1. Get `model.pt` (and ideally `model_best.pt`, ~280MB each) from whoever trained it — a shared
   drive / cloud storage link, not git.
2. Place it at `backend/vectorstore/model.pt`.
3. Skip straight to **Step 4 (run it)** — you do **not** need `movies.csv` or `ratings.csv` to
   serve the app. Everything the backend needs to answer requests (the movie ID mapping, titles,
   genre vocabulary) is baked into the checkpoint itself at training time; the raw dataset is only
   read during training.

## 3. Or: train from scratch (only if you don't have a checkpoint to reuse)

1. Download `ml-latest.zip` from https://grouplens.org/datasets/movielens/ (the full dataset, not
   `ml-latest-small`).
2. From the zip, you only need **`movies.csv`** and **`ratings.csv`** — the other files it ships
   (`genome-scores.csv` ~500MB, `genome-tags.csv`, `links.csv`, `tags.csv`) aren't read by any code
   path in this project; skip extracting them to save time and disk space.
3. Place both files at:
   ```
   backend/app/database/ml-latest/movies.csv
   backend/app/database/ml-latest/ratings.csv
   ```
4. Convert `ratings.csv` to the binary format the training pipeline streams from:
   ```bash
   cd backend
   venv\Scripts\python.exe convert_to_binary.py
   ```
   (writes `backend/app/database/ml-latest/ratings.npy`, required before training — training will
   fail with a missing-file error without this step.)
5. Train:
   ```bash
   venv\Scripts\python.exe train.py --top-n 20 --like-threshold 3.5
   ```
   Default is 10 epochs; add `--epochs N` to change it. Budget real time — roughly 8 min/epoch
   on GPU, longer on CPU. This writes `backend/vectorstore/model.pt` and `model_best.pt` when done.

## 4. Run it

```bash
# Backend (from backend/, venv activated)
uvicorn main:app --reload
# -> http://localhost:8000 (docs at /docs)
```

On startup, the backend loads `vectorstore/model.pt` if it exists (prints "Loading saved model…"),
or trains a fresh one from `ratings.csv` if it doesn't (this is the same as step 3.5, just
triggered automatically — slower than running `train.py` directly since it uses a smaller default
batch size, so prefer training via `train.py` explicitly if you need a checkpoint).

```bash
# Frontend, separate terminal, from frontend/
npm run dev
# -> http://localhost:5173
```

Open http://localhost:5173, register an account, rate a few movies, and click through
Rate → Build My Profile → Edit Profile → Get Recommendations.

**CORS note:** the backend only allows `http://localhost:5173` and `:5174` — keep the frontend on
Vite's default port.

## Gotchas

- No `.env` file or API keys are needed anywhere — auth uses a dev-default JWT secret
  (`backend/app/config/config.py`), fine for local use, not for a real deployment.
- The database is a local SQLite file, created automatically on first run
  (`backend/app/database/app.db`) — nothing to install separately there.
- No automated test suite exists yet — verifying a fresh setup is manual: register, rate movies,
  check the profile/recommendations/explanations render sensibly in the UI.
