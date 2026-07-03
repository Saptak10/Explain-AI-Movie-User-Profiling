# Full-Stack Implementation: Backend API + Frontend UI

This document describes every file that was created or modified to wire the AI layer into
a running full-stack application with user authentication, movie ratings, a profile page,
and an explainable recommendations page.

> **Last updated: 2026-07-01**
> Includes: A/B version assignment (Version O / Version N), counterbalanced 2-round
> experiment flow, per-movie and profile-level XAI editing, data logging for weight
> changes per round, SUS questionnaire with demographic pre-questions, streaming ML
> pipeline, new HCAI model architecture, and movie-level XAI explanations.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Tech Stack Decisions](#2-tech-stack-decisions)
3. [Experiment Design — Version O vs Version N](#3-experiment-design--version-o-vs-version-n)
4. [AI Layer — Streaming Pipeline & New Modules](#4-ai-layer--streaming-pipeline--new-modules)
5. [Backend — File-by-File](#5-backend--file-by-file)
   - [main.py (modified)](#mainpy-modified)
   - [config/config.py](#configconfigpy)
   - [database/db.py](#databasedbpy)
   - [schemas/user\_schema.py (modified)](#schemasuser_schemapy-modified)
   - [schemas/sus\_schema.py](#schemassus_schemapy)
   - [schemas/ai\_schema.py](#schemasai_schemapy)
   - [utils/jwt\_utils.py](#utilsjwt_utilspy)
   - [services/auth\_service.py](#servicesauth_servicepy)
   - [services/ai\_service.py](#servicesai_servicepy)
   - [routes/auth\_routes.py](#routesauth_routespy)
   - [routes/movie\_routes.py](#routesmovie_routespy)
   - [routes/ai\_routes.py](#routesai_routespy)
   - [routes/sus\_routes.py](#routessus_routespy)
5. [Frontend — File-by-File](#5-frontend--file-by-file)
   - [main.jsx (modified)](#mainjsx-modified)
   - [App.jsx (modified)](#appjsx-modified)
   - [index.css (modified)](#indexcss-modified)
   - [.env](#env)
   - [context/AuthContext.jsx](#contextauthcontextjsx)
   - [services/api.js](#servicesapijs)
   - [components/Navbar.jsx](#componentsnavbarjsx)
   - [components/StarRating.jsx](#componentsstarratingjsx)
   - [pages/LoginPage.jsx](#pagesloginpagejsx)
   - [pages/RatingsPage.jsx](#pagesratingspagejsx)
   - [pages/ProfilePage.jsx](#pagesprofilepagejsx)
   - [pages/RecommendPage.jsx](#pagesrecommendpagejsx)
   - [pages/SUSPage.jsx](#pagessuspagejsx)
6. [API Endpoint Reference](#6-api-endpoint-reference)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [How to Run](#8-how-to-run)
9. [Dependencies Added](#9-dependencies-added)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Browser  (http://localhost:5173)                            │
│                                                              │
│  LoginPage → RatingsPage → ProfilePage → RecommendPage      │
│                  │               │              │            │
│              StarRating     GenreSliders    OverridePanel    │
│                              ExplainModal                    │
└──────────────────────────┬──────────────────────────────────┘
                           │ Axios + JWT Bearer token
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI  (http://localhost:8000)                            │
│                                                              │
│  /api/auth/*   /api/movies/*   /api/profile                  │
│  /api/ratings  /api/recommend  /api/explain                  │
│                                                              │
│  auth_service ──► SQLite (users, ratings)                    │
│  ai_service   ──► SoftRegularizedHCAIAutoEncoder             │
│                   ├── get_profile()                          │
│                   ├── get_recommendations()                  │
│                   ├── get_recommendations_from_profile()     │
│                   └── explain_movie()                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  app/ai/ai.py  (teammate's AI implementation, unchanged)     │
│  MovieLens ml-latest-small dataset (100k ratings)            │
│  vectorstore/model.pt  (saved checkpoint, created on run)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Tech Stack Decisions

(unchanged — see table below)


| Decision | Choice | Reason |
|---|---|---|
| Database | SQLite (built-in `sqlite3`) | No MongoDB server needed; zero setup; sufficient for a demo |
| Password hashing | `hashlib.sha256` + `secrets` salt | Avoids `bcrypt`/`passlib` dependency issues on Python 3.13+ |
| Auth tokens | JWT via `python-jose` | Already in the venv; stateless, works with React |
| Model persistence | `torch.save` / `torch.load` | Train once (~2 min), reload instantly on every subsequent server start |
| AI inference in async context | `asyncio.to_thread()` | PyTorch is synchronous/CPU-bound; wrapping prevents blocking FastAPI's event loop |
| Frontend routing | `react-router-dom` | Standard SPA routing; enables protected routes and URL-based navigation |
| HTTP client | `axios` | Interceptors allow attaching JWT to every request automatically |
| Styling | Plain CSS with custom properties | No build-time dependency; fast to load; easy to understand |

---

## 3. Experiment Design — Version O vs Version N

The system implements a **between-subjects A/B experiment** to compare transparent versus opaque AI recommendations.

### Assignment — two layers of randomisation

When a user registers, `auth_service.register()` performs two random assignments:

1. **Version** (`users.version`): `random.choice(["O", "N"])` — which condition the user is in.
2. **Edit order** (`users.edit_order`): for Version O only, `random.choice(["movie_first", "profile_first"])` — which editing type the user encounters first in their two-round flow. Version N users get `edit_order = NULL`.

Both values are returned in the JWT login/register response and stored in `localStorage` via `AuthContext` (as `user.version` and `user.edit_order`). Assignments are permanent for the lifetime of the account.

### Version O — Transparent AI (2-Round Flow)

| Property | Detail |
|---|---|
| Badge | Green pill: **Version O — Transparent AI** + **Round X of 2** |
| Guidance text | Condition description shown in a left-bordered banner |
| Task instruction | Info banner describes what the user must do this round |
| Edit types | Per-movie weight editing (via "Edit Preferences ▼" card panel) and whole-profile editing (via a genre-slider panel at the top of the page) |
| Counterbalance | `edit_order = 'movie_first'` → round 1 uses per-movie editing, round 2 uses profile editing; `edit_order = 'profile_first'` → reversed. Eliminates order effects in analysis. |
| Gate per round | **Yes** — both `roundEdited` (edit applied) AND `hasRatedAny` (at least 1 recommended movie rated) must be true. |
| Round 1 button | "Next Round →" — resets edit/rating state and flips edit type; does NOT fetch new recs (the new recs are already shown after the last Apply click). |
| Round 2 button | "Continue to Survey →" — navigates to `/sus`. |

**Full user flow (Version O):**

```
Register (assigned version=O, edit_order=movie_first or profile_first)
  ↓
Rate up to 10 movies from the popular list (round=0 tagged in DB)
  ↓
View AI profile → Get Recommendations (round 1)
  ↓
Round 1 — rated_in_round=1 + edit type A (per assignment)
  • Rate at least one recommended movie (StarRating on each card, saved with round=1)
  • Apply at least one genre weight change using the assigned edit type
  • "Next Round →" unlocks
  ↓
Round 2 — rated_in_round=2 + edit type B (the other type)
  • Rate at least one recommended movie (saved with round=2)
  • Apply at least one genre weight change using the other edit type
  • "Continue to Survey →" unlocks
  ↓
SUS questionnaire (10 questions + 3 demographic questions)
```

### Version N — Standard AI

| Property | Detail |
|---|---|
| Badge | Grey pill: **Version N — Standard AI** |
| Guidance text | "You are in the Standard AI condition..." |
| Edit Preferences button | **Hidden** |
| Rating on rec cards | Shown (data collected), but not gated |
| Gate before survey | **None** — "Continue to Survey" always enabled |

**What the user does in Version N:**
1. Rates up to 10 movies
2. Views recommendations (with star ratings available on cards for data collection)
3. Clicks "Continue to Survey" directly

### Data recorded per user (linkable via `user_id`)

| Table | What it stores |
|---|---|
| `ratings` | Every movie rating, tagged with `round` (0=initial, 1=after round-1 recs, 2=after round-2 recs) |
| `profile_edits` | Every genre weight change applied: `round`, `edit_type` ('movie'/'profile'), `genre`, `level` (e.g. 'leicht verstärken'), `movie_id` (null for profile edits), `created_at` |
| `sus_responses` | SUS answers (10 questions × user) |
| `demographics` | Age group, degree/job, Netflix experience |
| `users` | `version`, `edit_order`, `has_edited` |

All tables share the `user_id` foreign key, so any user's complete interaction trace can be reconstructed with a simple join.

### Why this design

The research hypothesis is that users exposed to transparent, editable AI recommendations (Version O) will report higher system usability (SUS score) and better understanding of recommendations than users in the black-box condition (Version N). The counterbalanced edit order within Version O allows the analysis to separate the effect of editing type from the effect of editing order.

---

## 4. AI Layer — Streaming Pipeline & New Modules

Commit `59308c3` ("Add streaming ML pipeline, HCAI model, XAI") by the teammate replaced the original `app/ai/ai.py` monolith with five focused modules. The previous implementation loaded the entire ratings matrix into RAM as a dense NumPy array (610 users × 9742 movies for ml-latest-small). The new implementation streams ml-latest (~33 M rows, ~330 k users) without ever materialising the full matrix.

### Why the rewrite was needed

The full ml-latest dataset has ~87 k movies and ~330 k users. A dense `(num_users, num_movies)` float32 matrix would require ~330 k × 87 k × 4 bytes ≈ **115 GB of RAM** — far beyond any reasonable machine. The new pipeline caps peak RAM at `O(batch_size × num_movies)` by streaming `ratings.csv` line-by-line and yielding one user vector at a time via a PyTorch `IterableDataset`.

### New modules (`backend/app/ai/`)

#### `data_pipeline.py`
Streaming data pipeline over `ratings.csv`.

Key classes:
- `RatingsStreamReader` — decodes each CSV row, translates `movieId` to dense index
- `UserAggregator` — buffers one user's rows; flushes a complete user vector when the userId changes. Requires (and validates) that `ratings.csv` is sorted by `userId` ascending — as the official MovieLens export always is.
- `SparseUserVectorDataset` — PyTorch `IterableDataset`; turns each flushed user into a `(input_vector, target_vector, hidden_mask)` triple for masked autoencoder training

#### `id_mapping.py`
`IdMapping` dataclass that owns the bidirectional movie ID translation and genre vocabulary. Built once from `movies.csv` by `build_id_mapping()`, then saved into the checkpoint so subsequent loads don't re-scan the file.

Key responsibilities:
- `movie_id_to_idx` / `idx_to_movie_id` — translate between real MovieLens IDs (non-contiguous, e.g. 193609) and dense 0-based indices
- `genres` list — dynamically discovered from `movies.csv` (not hardcoded)
- `genre_mask` — `(num_movies, num_genres)` binary prior-knowledge matrix used for weight clipping

#### `model.py`
`DualModeHCAIAutoEncoder` — the genre-bottleneck autoencoder, refactored to accept an `IdMapping` for dynamic sizing.

Two named forward passes replace the old implicit `user_overrides` branch:
- `forward_standard(x)` → `(predictions, latent_profile)` — standard encoder → bottleneck → decoder pass
- `forward_interactive(genre_override_vec)` → `predictions` — skips the encoder entirely; uses the provided genre weight vector directly as the bottleneck, then decodes to recommendation scores

Additional helpers:
- `extract_taste_profile(latent, genres)` → `{genre: float}` dict for the `/api/profile` endpoint
- `clip_encoder_weights()` — called after each training step to anchor encoder weights near the genre prior

#### `losses.py`
- `masked_mse_loss(pred, target, mask)` — only penalises positions where the user has actually rated, preventing the model from overfitting to structural zeros
- `train_step(model, optimizer, input, target, mask, lambda_reg, epsilon_clip)` — one mini-batch update: forward, masked MSE + L2 regularisation, backward, weight clipping

#### `xai.py`
Three Human-Centered XAI utilities:

| Function | Purpose |
|---|---|
| `hydrate_sparse_input(ratings, id_mapping)` | Converts sparse `{movieId: rating}` API payload → dense `(1, num_movies)` tensor. The only place a dense input vector is constructed per request. |
| `compute_local_feature_importance(model, vec, target_idx)` | Permutation-importance restricted to the user's *non-zero* ratings only. Zeroing already-zero entries changes nothing; scanning all 87 k movies at inference time would cause API timeouts. Returns `{dense_idx: importance_float}` for the user's actual watch history. |
| `generate_soft_rationale(vec, latent_profile, target_movie_idx, id_mapping, genre_mask_row)` | Cross-references user's top-rated movies, dominant bottleneck genre nodes, and the target movie's genre membership to produce a natural-language rationale string. |

### What changed in the public AI service API

| Method | Old signature | New signature |
|---|---|---|
| `setup()` | `(model_save_path)` | `(model_save_path, movies_csv_path, ratings_csv_path)` |
| `genres` | constant list from `ai.py` | property on `AIService` (from `id_mapping.genres`) |
| `get_recommendations()` | `(ratings, top_n, overrides, alpha)` | `(ratings, top_n)` — no overrides |
| `get_recommendations_from_profile()` | `(edited_dict, ratings, top_n)` — 0–1 float values, old format | `(genre_overrides, ratings, top_n)` — `{genre_name: float}` genre weight dict, passed directly to `forward_interactive` |
| `explain_movie()` | `(movie_idx, ratings, method)` — returned `{contributions, text}` | `(movie_id, ratings)` — returns `{movie_id, title, rationale, feature_importance}` where `feature_importance` is movie-level, not genre-level |

### Frontend impact

Because `get_recommendations()` no longer accepts overrides, both the Profile page and Recommendation page override systems now convert the user's 5-button selections (−−/−/○/+/++) to float genre weights and call `POST /api/recommend/edited-profile` instead of `POST /api/recommend`:

```
User presses −− on Drama (AI profile: 0.85)
  → delta = −0.5
  → genre_weight["Drama"] = max(0, min(1, 0.85 − 0.5)) = 0.35
  → POST /api/recommend/edited-profile { genre_weights: {"Drama": 0.35, ...} }
  → ai_service.get_recommendations_from_profile({"Drama": 0.35, ...})
  → model.forward_interactive(override_vec)
  → updated recommendation list
```

The explanation panel in the Recommendation page now shows:
1. **`rationale`** — the natural-language reason string from `generate_soft_rationale()`
2. **`feature_importance`** — top-5 movies from the user's history that most influenced this recommendation (with importance bars), from `compute_local_feature_importance()`

---

## 5. Backend — File-by-File

### `main.py` (modified)

**Location:** `backend/main.py`

**Before:** 5 lines — a FastAPI instance with a single `GET /` health check. Nothing was wired up.

**After:** The real application entry point. Responsibilities:

1. **CORS middleware** — allows `http://localhost:5173` and `5174` (Vite's default ports) to call the backend. Without this, browsers block cross-origin requests.

2. **Startup event** — runs once when the server starts:
   - Calls `db.init_db()` to create the SQLite file and tables if they don't exist
   - Checks whether `vectorstore/model.pt` already exists
     - If **yes**: loads the saved checkpoint (instant)
     - If **no**: trains the model on MovieLens data (~1-2 minutes), then saves to disk
   - Both operations are wrapped in `asyncio.to_thread()` so they don't block startup

3. **Router registration** — mounts the three routers:
   ```
   /api/auth/*    ← auth_router
   /api/movies/*  ← movie_router
   /api/*         ← ai_router
   ```

---

### `config/config.py`

**Location:** `backend/app/config/config.py`

**Before:** Empty (1-line placeholder).

Typed settings class using `pydantic-settings`. Reads values from `backend/.env` if present, otherwise uses safe defaults that work out of the box for development.

| Setting | Default | Used by |
|---|---|---|
| `jwt_secret` | `"dev-secret-change-in-production"` | JWT signing/verification |
| `jwt_algorithm` | `"HS256"` | JWT |
| `jwt_expiry_hours` | `24` | Token lifespan |
| `db_path` | `"app/database/app.db"` | SQLite file location |
| `model_save_path` | `"vectorstore/model.pt"` | PyTorch checkpoint |

Imported anywhere as:
```python
from app.config.config import settings
```

---

### `database/db.py`

**Location:** `backend/app/database/db.py`

**Before:** Empty.

A thin async wrapper around Python's built-in `sqlite3`. Since `sqlite3` is synchronous but FastAPI runs on an async event loop, every database operation is offloaded to a thread pool via `asyncio.to_thread()`.

**Database schema created by `init_db()`:**

```sql
CREATE TABLE users (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    username         TEXT UNIQUE NOT NULL,
    hashed_password  TEXT NOT NULL,
    has_edited       INTEGER NOT NULL DEFAULT 0,  -- 1 after first Apply in Version O
    sus_done         INTEGER NOT NULL DEFAULT 0,  -- 1 after SUS submission
    version          TEXT    NOT NULL DEFAULT 'O', -- 'O' (Transparent) or 'N' (Standard)
    edit_order       TEXT                          -- 'movie_first' | 'profile_first' | NULL (N users)
);

CREATE TABLE ratings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER NOT NULL,
    movie_id  INTEGER NOT NULL,
    rating    REAL    NOT NULL,
    round     INTEGER NOT NULL DEFAULT 0, -- 0=initial ratings, 1=round-1 recs, 2=round-2 recs
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, movie_id)  -- re-rating replaces old value (round also updated)
);

CREATE TABLE profile_edits (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    round      INTEGER NOT NULL,            -- 1 or 2
    edit_type  TEXT    NOT NULL,            -- 'movie' | 'profile'
    genre      TEXT    NOT NULL,            -- e.g. 'Action'
    level      TEXT    NOT NULL,            -- e.g. 'leicht verstärken'
    movie_id   INTEGER,                     -- set for edit_type='movie', NULL for 'profile'
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE sus_responses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL,
    question_idx INTEGER NOT NULL,  -- 0..9
    response     INTEGER NOT NULL,  -- 1..5
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, question_idx)
);

CREATE TABLE demographics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER NOT NULL,
    age_group           TEXT,     -- '18-23' | '24-30' | '30-45' | '>45'
    degree_job          TEXT,     -- free text: degree or job title
    netflix_experience  INTEGER,  -- 1..5 Likert
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id)
);
```

The `init_db()` function also handles **schema migration** for existing databases: it uses `ALTER TABLE … ADD COLUMN` with try/except for each new column, so the server can be restarted without wiping existing user data.

**Public functions:**

| Function | Returns | Used for |
|---|---|---|
| `init_db(path)` | — | Called once at startup |
| `execute(sql, params)` | `int` (last row ID) | INSERT, UPDATE, DELETE |
| `fetchone(sql, params)` | `dict \| None` | SELECT one row |
| `fetchall(sql, params)` | `list[dict]` | SELECT multiple rows |

---

### `schemas/user_schema.py` (modified)

Two Pydantic models for the auth API:

- `AuthRequest` — request body for both register and login: `{username: str, password: str}`
- `TokenResponse` — what the server returns: `{token: str, user_id: int, username: str, version: str, edit_order: str | None}`

The `version` field (`"O"` or `"N"`) is assigned at registration and returned on every login. `edit_order` (`"movie_first"`, `"profile_first"`, or `null` for Version N) controls the round-1 / round-2 editing sequence in the frontend.

---

### `schemas/sus_schema.py`

**Location:** `backend/app/schemas/sus_schema.py`

Pydantic model for the combined SUS + demographics submission:

```python
class SUSRequest(BaseModel):
    responses:          list[int]  # exactly 10 values, each 1–5
    age_group:          str        # '18-23' | '24-30' | '30-45' | '>45'
    degree_job:         str        # free text
    netflix_experience: int        # 1–5

    @field_validator("responses")  → exactly 10 items, each in 1..5
    @field_validator("age_group")  → must be one of the four valid options
    @field_validator("netflix_experience") → must be in 1..5
```

All three demographic fields are required — the submit button is disabled in the frontend until they are all filled in.

---

### `schemas/ai_schema.py`

**Location:** `backend/app/schemas/ai_schema.py` (new file)

Four Pydantic models for the AI endpoints:

```python
class RatingRequest:
    movie_id: int    # dense model index shown in the UI
    rating:   float  # 1.0–5.0
    round:    int    # 0=initial, 1=round-1 recs, 2=round-2 recs (default 0)

class ProfileEditLogRequest:
    round:     int          # 1 or 2
    edit_type: str          # 'movie' | 'profile'
    genre:     str          # e.g. 'Action'
    level:     str          # e.g. 'leicht verstärken'
    movie_id:  int | None   # set for movie edits, null for profile edits

class RecommendRequest:
    top_n:     int  = 10
    overrides: dict = None  # {"Sci-Fi": "stark verstärken", "Comedy": "stark dämpfen"}
    alpha:     float = 3.0  # strength of the override effect

class EditedProfileRequest:
    profile: dict  # {"Drama": 0.85, "Comedy": 0.12, ...}  values 0–1
    top_n:   int = 10

class ExplainRequest:
    movie_id: int
    method:   str = "soft"  # "soft" (fast) or "lime" (requires sklearn)
```

---

### `utils/jwt_utils.py`

**Location:** `backend/app/utils/jwt_utils.py` (new file)

Two items:

**`create_token(user_id)`** — encodes the user's integer ID into a signed JWT with a 24-hour expiry. Called after successful register/login.

**`get_current_user`** — a FastAPI dependency. Any route that declares:
```python
async def my_route(user_id: int = Depends(get_current_user)):
```
...will automatically:
- Read the `Authorization: Bearer <token>` header
- Verify the signature and expiry
- Return the `user_id` integer on success
- Return HTTP 401 if the token is missing, invalid, or expired

Uses `python-jose` (already installed in the venv).

---

### `services/auth_service.py`

**Location:** `backend/app/services/auth_service.py` (new file)

Business logic for user registration and login. Deliberately avoids `passlib`/`bcrypt` because Python 3.13 removed the `crypt` standard library module that older passlib versions depend on. Instead uses only built-in modules:

**Password storage format:** `"<16-byte-hex-salt>:<sha256(salt+password)>"`

```python
def _hash(password)  → "a3f8...bc12:e9d1..."
def _verify(password, stored) → bool
```

**`register(username, password)`:**
1. Checks if the username already exists — raises HTTP 400 if so
2. Hashes the password
3. Calls `random.choice(["O", "N"])` to assign the experiment version
4. Inserts into the `users` table with the assigned version
5. Returns `{id, username, version}`

**`login(username, password)`:**
1. Looks up the user by username
2. Verifies the password hash — raises HTTP 401 if wrong
3. Returns `{id, username, version}` — the stored version is read from the DB so it is consistent across sessions

---

### `services/ai_service.py`

**Location:** `backend/app/services/ai_service.py` (new file)

The central piece of the backend. Wraps the teammate's standalone `ai.py` script into a long-lived service object that loads once at startup and serves inference requests.

A module-level singleton is exported:
```python
ai_service = AIService()
```

#### `setup(save_path)` and `train_and_save()`

`train_and_save()` replicates the `__main__` block from `ai.py` but saves a portable checkpoint:

1. Loads MovieLens CSVs via `load_movielens_data()`
2. Builds the dense `movie_id → index` mapping (MovieLens IDs are not contiguous, so this is necessary)
3. Builds the 610×9742 rating matrix `R` and the 9742×18 genre matrix
4. Creates and trains `SoftRegularizedHCAIAutoEncoder` for 25 epochs
5. Saves to `vectorstore/model.pt`:
   ```
   {state_dict, movie_id_to_idx, idx_to_title, num_movies, genre_np, R_matrix}
   ```

#### `load()`

Reads the checkpoint and reconstructs the model in a single call. This is what runs on every server restart after the first.

#### `_init()` — called by both train and load

Pre-computes and caches everything the API routes need:

- **`all_latent_profiles`** — runs all 610 training users through the encoder once; stored for potential future use
- **`popular_movies`** — the 50 movies with the most ratings in the dataset; these are shown on the Rate Movies page
- **`global_importance`** — approximated from encoder weight norms; returned by `GET /api/importance`

#### Inference methods

| Method | What it does |
|---|---|
| `_user_vec(ratings)` | Converts `{"42": 4.0, "108": 3.0}` into a sparse `(1, 9742)` PyTorch tensor |
| `get_profile(ratings)` | Runs the encoder → returns `{"Drama": 0.78, "Sci-Fi": 0.61, ...}` |
| `get_recommendations(ratings, overrides, alpha)` | Runs the full forward pass (with optional override tensor), excludes already-rated movies, returns top N |
| `get_recommendations_from_profile(edited, ratings)` | Bypasses the encoder entirely — passes the human-edited profile directly to the decoder (`recommend_from_edited_profile`) |
| `explain_movie(movie_idx, ratings, method)` | Returns XAI explanation: either "soft" (weight × profile per genre) or "lime" (local linear approximation via sklearn Ridge) |

---

### `routes/auth_routes.py`

**Location:** `backend/app/routes/auth_routes.py`

Two endpoints mounted at `/api/auth`:

```
POST /api/auth/register   body: {username, password}  → {token, user_id, username, version}
POST /api/auth/login      body: {username, password}  → {token, user_id, username, version}
```

Both call the corresponding `auth_service` function and then `create_token()` to issue a JWT. The `version` field (`"O"` or `"N"`) is now included in every auth response.

---

### `routes/sus_routes.py`

**Location:** `backend/app/routes/sus_routes.py`

Four endpoints mounted at `/api/sus`:

| Method | Path | Auth | Request body | Response |
|---|---|---|---|---|
| GET | `/questions` | None | — | `{questions: [string × 10]}` |
| POST | `/submit` | Bearer | `SUSRequest` | `{done: true}` |
| GET | `/results` | Bearer | — | `{score: float 0–100}` |

**`POST /sus/submit` saves three things atomically:**
1. All 10 SUS responses into `sus_responses` (INSERT OR REPLACE per question index)
2. Demographic data into `demographics` (INSERT OR REPLACE for the user)
3. Sets `users.sus_done = 1`

**SUS scoring algorithm** (standard Brooke 1996):
- Odd-indexed questions (0, 2, 4, 6, 8): contribution = `response − 1`
- Even-indexed questions (1, 3, 5, 7, 9): contribution = `5 − response`
- Total = sum of all contributions × 2.5
- Result range: 0 (worst) to 100 (best)

---

### `routes/movie_routes.py`

**Location:** `backend/app/routes/movie_routes.py` (new file)

One endpoint mounted at `/api/movies`:

```
GET /api/movies/popular   → {movies: [{id, title}, ...]}   (50 most-rated movies)
```

No authentication required. Used by the Rate Movies page on load to know which movies to display.

---

### `routes/ai_routes.py`

**Location:** `backend/app/routes/ai_routes.py` (new file)

All AI endpoints mounted at `/api`. Every endpoint is protected by JWT — unauthenticated requests receive HTTP 401.

All calls to `ai_service` are wrapped in `asyncio.to_thread()` to prevent PyTorch's synchronous operations from blocking FastAPI's event loop.

| Method | Path | Body / Params | Returns |
|---|---|---|---|
| `GET` | `/api/genres` | — | `{genres: [18 genre names]}` |
| `POST` | `/api/ratings` | `{movie_id, rating}` | `{status: "ok"}` |
| `GET` | `/api/ratings` | — | `{ratings: {movie_id: rating}}` |
| `GET` | `/api/profile` | — | `{profile: {genre: float}}` |
| `POST` | `/api/recommend` | `{top_n, overrides?, alpha}` | `{recommendations: [{movie_id, title, score}]}` |
| `POST` | `/api/recommend/edited-profile` | `{profile, top_n}` | `{recommendations: [...]}` |
| `POST` | `/api/explain` | `{movie_id, method}` | `{method, text?, contributions: [{genre, value}]}` |
| `GET` | `/api/importance` | — | `{importance: {genre: float}}` |

---

## 5. Frontend — File-by-File

### `main.jsx` (modified)

**Before:** Rendered `<App />` in a `StrictMode` wrapper — nothing else.

**After:** Wraps the entire app in two providers that all child components need:

```jsx
<StrictMode>
  <AuthProvider>      ← makes user state available everywhere
    <BrowserRouter>   ← enables URL-based navigation
      <App />
    </BrowserRouter>
  </AuthProvider>
</StrictMode>
```

---

### `App.jsx` (modified)

**Before:** The default Vite scaffold (counter button, links to React and Vite documentation).

**After:** Defines the four URL routes and a `<Guard>` component:

```jsx
function Guard({ children }) {
  const { user } = useAuth()
  return user ? children : <Navigate to="/login" replace />
}
```

Route table:

| URL | Component | Protected |
|---|---|---|
| `/login` | `LoginPage` | No — redirects to `/rate` if already logged in |
| `/rate` | `RatingsPage` | Yes |
| `/profile` | `ProfilePage` | Yes |
| `/recommend` | `RecommendPage` | Yes |
| `*` | Redirects | Based on auth state |

---

### `index.css` (modified)

**Before:** Vite scaffold styles — constrained `#root` width, counter component, dark mode variables.

**After:** Complete application stylesheet (~350 lines). Organised into sections:

| Section | Key classes |
|---|---|
| Design tokens | CSS custom properties for color, shadow, radius |
| Navbar | `.navbar`, `.nav-link`, `.nav-brand`, `.nav-user` |
| Buttons | `.btn-primary`, `.btn-secondary`, `.btn-ghost`, `.btn-full` |
| Auth page | `.auth-container`, `.auth-card`, `.tab-group`, `.field` |
| Movie grid | `.movie-grid`, `.movie-card`, `.movie-card.rated` |
| Star rating | `.stars`, `.star`, `.star.filled` |
| Profile bars | `.genre-row`, `.bar-track`, `.bar-fill`, `.genre-slider` |
| Rec cards | `.rec-card`, `.rec-rank`, `.score-track`, `.score-fill` |
| Override panel | `.override-panel`, `.override-row`, `.override-btn.active` |
| Explain modal | `.modal-overlay`, `.modal`, `.explain-fill.pos`, `.explain-fill.neg` |

**Color palette:**

| Token | Value | Used for |
|---|---|---|
| `--primary` | `#4f46e5` | Buttons, active states, score bars |
| `--primary-light` | `#eef2ff` | Hover backgrounds, rated-card background |
| `--bg` | `#f0f2f5` | Page background |
| `--card` | `#ffffff` | Card backgrounds |
| `--star` | `#f59e0b` | Filled stars |
| `--pos` | `#4f46e5` | Positive genre contributions in explain modal |
| `--neg` | `#ef4444` | Negative genre contributions in explain modal |

---

### `.env`

**Location:** `frontend/.env`

One line: `VITE_API_URL=http://localhost:8000`

Vite reads this at build time. All frontend code accesses it as `import.meta.env.VITE_API_URL`. The `api.js` service uses it as the Axios base URL.

---

### `context/AuthContext.jsx`

**Location:** `frontend/src/context/AuthContext.jsx` (new file and directory)

React context that stores the currently logged-in user (`{id, username}`) and the `login`/`logout` actions.

Persists to `localStorage` so a browser refresh does not log the user out:

- `login(token, user_id, username, version)` → writes to `localStorage` as `{id, username, version}`, updates state
- `logout()` → clears `localStorage`, sets user to `null`
- `useAuth()` hook → any component can call this to read `user.version` or trigger login/logout

---

### `services/api.js`

**Location:** `frontend/src/services/api.js` (new file and directory)

A single pre-configured Axios instance. A **request interceptor** reads the JWT from `localStorage` and attaches it as an `Authorization: Bearer <token>` header to every outgoing request automatically — no component needs to handle auth headers manually.

Three grouped export objects:

```javascript
authApi   { register, login }
moviesApi { popular }
aiApi     { submitRating, getRatings, getProfile, recommend,
            recommendFromProfile, explain, getImportance }
```

---

### `components/Navbar.jsx`

**Location:** `frontend/src/components/Navbar.jsx` (new file and directory)

Sticky top bar (60px height, white background, bottom border).

- **Left:** brand link `🎬 CineProfile`
- **Centre:** three nav links — Rate Movies, My Profile, Recommendations (only visible when logged in). Active link detected via `useLocation()`.
- **Right:** username display + Logout button (calls `AuthContext.logout()` and redirects to `/login`)

---

### `components/StarRating.jsx`

**Location:** `frontend/src/components/StarRating.jsx` (new file)

Reusable 5-star rating component. Each `★` is a button:

- Filled stars (value ≥ star number) are amber (`#f59e0b`)
- Empty stars are gray
- Hovering scales the star up slightly
- Clicking sets the rating; clicking the same star again keeps the same value
- Shows the numeric value (e.g. `3.0`) to the right when a rating is set

Used by `RatingsPage` for every movie card.

---

### `pages/LoginPage.jsx`

**Location:** `frontend/src/pages/LoginPage.jsx` (new file and directory)

Centered card with a tab switcher between Sign In and Register. Both modes use the same two-field form (username + password) — the only difference is which `authApi` function is called.

On success: saves the returned JWT via `AuthContext.login()` and navigates to `/rate`.
On failure: displays the server's `detail` error message below the form.

---

### `pages/RatingsPage.jsx`

**Location:** `frontend/src/pages/RatingsPage.jsx` (new file)

Loads two things in parallel on mount:
1. `GET /api/movies/popular` — the 50 most-rated movies to display
2. `GET /api/ratings` — the user's existing ratings (pre-fills stars if they've rated before)

**Interaction flow:**
- User clicks stars to set a rating for each movie
- `movie-card.rated` CSS class is applied (card turns indigo-tinted) when any star is selected
- A sticky CTA bar appears at the bottom once at least one movie is rated
- "Build My Profile →" submits only changed/new ratings (not all 50) via `POST /api/ratings`, then navigates to `/profile`

---

### `pages/ProfilePage.jsx`

**Location:** `frontend/src/pages/ProfilePage.jsx` (new file)

Calls `GET /api/profile` on mount, which runs the autoencoder's encoder on the user's saved ratings and returns 18 genre affinities.

**Two display modes:**

| Mode | UI | Behaviour |
|---|---|---|
| AI Profile (default) | Read-only horizontal bars, sorted by value descending | Shows what the model inferred about the user |
| Edit Profile (toggle) | Range sliders 0–100% | User can override any genre weight manually |

The top genre name is shown in the subtitle in AI mode.

"Get Recommendations →" behaviour differs by mode:
- **AI mode** → navigates to `/recommend`, which fetches recommendations from the server on arrival
- **Edit mode** → calls `POST /api/recommend/edited-profile` with the slider values immediately, passes results as React Router state to `/recommend` so no second API call is needed

---

### `pages/RecommendPage.jsx`

**Location:** `frontend/src/pages/RecommendPage.jsx`

The most interactive page. Full-width single-column list of recommendation cards.

**Version-conditional rendering** — reads `user.version` from `AuthContext`:

| Element | Version O (Transparent) | Version N (Standard) |
|---|---|---|
| Version badge | Green "Version O — Transparent AI" | Grey "Version N — Standard AI" |
| Guidance banner | "You can see why each movie was recommended and adjust genre weights…" | "You receive AI-generated recommendations without explanations…" |
| Task banner | "Edit at least one movie's preferences before continuing" (hidden after first Apply) | Not shown |
| Edit Preferences button | Shown on every card (`showEdit=true`) | Hidden (`showEdit=false`) |
| XAI expand panel | Available on expand | Not shown |
| Continue to Survey | Disabled until `hasEdited=true` | Always enabled |

**`RecCard` component:**

Each card shows rank, movie title, and score bar. When `showEdit=true` (Version O only), an "Edit Preferences ▼ / Close ▲" toggle button appears.

**`EditPanel` component (Version O only):**

Loaded asynchronously from `POST /api/explain` when a card is first expanded. Displays:
- Natural-language explanation text from the AI
- Top-5 genre contributions as mini bar chart
- 5-level override buttons per genre (−−, −, ○, +, ++)
- "Apply & Refresh Recommendations" button

On first Apply: sets `hasEdited=true` in React state + calls `POST /api/user/mark-edited` to persist to DB.

---

### `pages/SUSPage.jsx`

**Location:** `frontend/src/pages/SUSPage.jsx`

Combined demographic questionnaire + System Usability Scale. Structured in two sections separated by a horizontal divider:

**Section 1 — Background Information (3 questions):**

| # | Question | Input type | Options / Range |
|---|---|---|---|
| 1 | How old are you? | Radio buttons (pill style) | 18-23 / 24-30 / 30-45 / >45 |
| 2 | Which degree are you currently taking or what is your job title? | Text input | Free text |
| 3 | I have experience with movie recommendation platforms such as Netflix | Likert 1–5 | 1 = Strongly Disagree, 5 = Strongly Agree |

All three are required. The submit button stays disabled until all demographic fields are completed **and** all 10 SUS questions are answered.

**Section 2 — System Usability Scale (10 questions):**

Standard Brooke (1996) SUS questionnaire, identical for both Version O and N:
1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

Each question uses circular radio buttons (1–5). Answered questions have their number circle turn from grey to indigo. Progress counter at the bottom.

**Submission:** `POST /api/sus/submit` with `{responses, age_group, degree_job, netflix_experience}`.

**Post-submission:** "Thank You!" completion screen with a "Back to Recommendations" button.

---

## 6. API Endpoint Reference

### Auth — `/api/auth`

| Method | Path | Auth | Request body | Response |
|---|---|---|---|---|
| POST | `/register` | None | `{username, password}` | `{token, user_id, username, version, edit_order}` |
| POST | `/login` | None | `{username, password}` | `{token, user_id, username, version, edit_order}` |

### Movies — `/api/movies`

| Method | Path | Auth | Response |
|---|---|---|---|
| GET | `/popular` | None | `{movies: [{id, title}]}` (50 items) |

### AI — `/api`

All endpoints below require `Authorization: Bearer <token>`.

| Method | Path | Request body | Response |
|---|---|---|---|
| GET | `/genres` | — | `{genres: [string]}` |
| POST | `/ratings` | `{movie_id: int, rating: float, round: int}` | `{status: "ok"}` |
| GET | `/ratings` | — | `{ratings: {movie_id: rating}}` |
| GET | `/profile` | — | `{profile: {genre: float 0–1}}` |
| POST | `/recommend` | `{top_n: int}` | `{recommendations: [{movie_id, title, score}]}` |
| POST | `/recommend/edited-profile` | `{genre_weights: {genre: float}, top_n: int}` | `{recommendations: [...]}` |
| POST | `/explain` | `{movie_id: int}` | `{movie_id, title, rationale: str, feature_importance: [{movie_id, title, importance}]}` |
| GET | `/importance` | — | `{importance: {genre: float}}` |
| POST | `/user/mark-edited` | — | `{status: "ok"}` |
| POST | `/profile-edits` | `{round, edit_type, genre, level, movie_id?}` | `{status: "ok"}` |

### SUS — `/api/sus`

| Method | Path | Auth | Request body | Response |
|---|---|---|---|---|
| GET | `/questions` | None | — | `{questions: [string × 10]}` |
| POST | `/submit` | Bearer | `{responses, age_group, degree_job, netflix_experience}` | `{done: true}` |
| GET | `/results` | Bearer | — | `{score: float 0–100}` |

---

## 7. Data Flow Diagrams

### Login / Registration

```
User registers:
    → POST /api/auth/register {username, password}
    → auth_service.register()
        → hash password
        → random.choice(["O","N"]) → version assigned
        → INSERT INTO users (username, hashed_password, version)
    → create_token(user_id) → JWT
    → {token, user_id, username, version} returned
    → AuthContext.login() stores {id, username, version} in localStorage
    → navigate to /rate

User logs in:
    → POST /api/auth/login {username, password}
    → auth_service.login() verifies sha256 hash
    → SELECT version FROM users → returns stored version
    → {token, user_id, username, version} returned
    → AuthContext.login() updates localStorage
    → navigate to /rate
```

### Rating movies and building a profile

```
RatingsPage loads
    → GET /api/movies/popular  → 50 most-rated movies from ai_service.popular_movies
    → GET /api/ratings         → user's existing ratings from SQLite

User clicks stars, then "Build My Profile"
    → POST /api/ratings × N    → saved to SQLite ratings table
    → navigate to /profile

ProfilePage loads
    → GET /api/profile
    → db.fetchall ratings for user
    → ai_service.get_profile(ratings)
        → _user_vec(): sparse (1×9742) tensor
        → model encoder: (1×9742) → (1×128) → (1×18) latent profile
        → {genre: float} returned
    → displayed as sorted bar chart
```

### Recommendations with override

```
User sets Sci-Fi → "++" and Comedy → "--", clicks Apply

    → POST /api/recommend {overrides: {"Sci-Fi": "stark verstärken", "Comedy": "stark dämpfen"}, alpha: 3.0}
    → ai_service.get_recommendations(ratings, overrides, alpha=3.0)
        → _user_vec(): sparse rating tensor
        → build_override_tensor({"Sci-Fi": "stark verstärken", ...}) → (1×18) float tensor
        → model.forward(x, user_overrides=tensor, alpha=3.0)
            Path A: encoder → latent → decoder → ann_output
            Path B: override_tensor @ target_genre_matrix → explicit_bias
            combined: ann_output + (alpha × explicit_bias)
        → sort scores, exclude rated movies, return top 10
    → recommendation list refreshes
```

### Explanation

```
User clicks "Why?" on a movie

    → POST /api/explain {movie_id: 42, method: "soft"}
    → ai_service.explain_movie(42, ratings, "soft")
        → encode user ratings → latent profile
        → generate_soft_xai_explanation(): checks core genres (weight > 0.85 AND affinity > 0.6)
        → compute genre contributions: encoder_l2 @ encoder_l1 → per-genre weight for film
          contribution[i] = weight[genre_i, movie_42] × user_profile[genre_i]
        → sort by |contribution|, return top 8
    → ExplainModal shows text + bar chart
```

### SUS + Demographics submission

```
SUSPage — user completes all fields:
    Demographics: age group (radio), degree/job (text), Netflix experience (1–5 Likert)
    SUS: 10 questions answered (1–5 each)
    → "Submit Survey" enabled

    → POST /api/sus/submit {responses:[...], age_group, degree_job, netflix_experience}
    → sus_routes.submit_sus()
        → INSERT OR REPLACE INTO sus_responses × 10 rows
        → INSERT OR REPLACE INTO demographics (age_group, degree_job, netflix_experience)
        → UPDATE users SET sus_done = 1
    → {done: true} → completion screen shown
```

---

## 8. How to Run

### Prerequisites

- Python venv with packages installed (`torch`, `fastapi`, `uvicorn`, `pandas`, `python-jose`, `pydantic-settings`)
- Node.js 18+ with `npm`

### Step 1 — Backend (Terminal 1)

```bash
# From project root
source backend/venv/bin/activate

cd backend
uvicorn main:app --reload
```

**First run only:** the terminal will print training progress:
```
First run — training model (~1-2 min)…
Loading MovieLens data…
Building matrices (610 users × 9742 movies)…
Training (25 epochs)…
  Epoch 5/25 done
  ...
  Epoch 25/25 done
Saving checkpoint…
AI service ready.
```

Every subsequent start prints:
```
Loading saved model…
AI service ready.
```

Interactive API docs available at: `http://localhost:8000/docs`

### Step 2 — Frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

App available at: `http://localhost:5173`

### User flow

1. Open `http://localhost:5173`
2. Register a new account (version O or N is randomly assigned)
3. Rate at least a few movies on the Rate Movies page
4. Click "Build My Profile →"
5. View your AI-inferred genre affinities on the Profile page
6. Optionally toggle "Edit Profile" and drag sliders to manually adjust
7. Click "Get Recommendations →"

**If Version O (Transparent):**
8. See version badge and guidance explaining the transparent condition
9. Click "Edit Preferences ▼" on any recommendation card
10. Review why it was recommended (genre contribution chart + text)
11. Adjust genre weights using −−/−/○/+/++ buttons
12. Click "Apply & Refresh Recommendations"
13. "Continue to Survey →" becomes active

**If Version N (Standard):**
8. See version badge and guidance explaining the standard condition
9. Review the recommendations list
10. Click "Continue to Survey →" (always enabled)

**Both versions:**
11. Answer 3 demographic questions (age, degree/job, Netflix experience)
12. Answer 10 SUS questions
13. Click "Submit Survey"
14. See "Thank You!" completion screen

---

## 9. Dependencies Added

### Backend — no new packages required

All packages used (`fastapi`, `uvicorn`, `torch`, `pandas`, `numpy`, `python-jose`, `pydantic-settings`) were already present in `backend/venv/`.

The implementation deliberately avoided new backend dependencies:
- `sqlite3` — Python built-in
- Password hashing — `hashlib` + `secrets` (built-in)
- No `bcrypt`, `passlib`, `motor`, or `sqlalchemy` needed

### Frontend — two packages added

```bash
npm install axios react-router-dom --save
```

| Package | Version | Purpose |
|---|---|---|
| `axios` | ^1.x | HTTP client with request interceptors for auto-attaching JWT |
| `react-router-dom` | ^7.x | URL routing, `<Routes>`, `<Navigate>`, `useNavigate`, `useLocation` |
