# Full-Stack Implementation: Backend API + Frontend UI

This document describes every file that was created or modified to wire the AI layer into
a running full-stack application with user authentication, movie ratings, a profile page,
and an explainable recommendations page.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Tech Stack Decisions](#2-tech-stack-decisions)
3. [Backend — File-by-File](#3-backend--file-by-file)
   - [main.py (modified)](#mainpy-modified)
   - [config/config.py](#configconfigpy)
   - [database/db.py](#databasedbpy)
   - [schemas/user\_schema.py (modified)](#schemasuser_schemapy-modified)
   - [schemas/ai\_schema.py](#schemasai_schemapy)
   - [utils/jwt\_utils.py](#utilsjwt_utilspy)
   - [services/auth\_service.py](#servicesauth_servicepy)
   - [services/ai\_service.py](#servicesai_servicepy)
   - [routes/auth\_routes.py](#routesauth_routespy)
   - [routes/movie\_routes.py](#routesmovie_routespy)
   - [routes/ai\_routes.py](#routesai_routespy)
4. [Frontend — File-by-File](#4-frontend--file-by-file)
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
5. [API Endpoint Reference](#5-api-endpoint-reference)
6. [Data Flow Diagrams](#6-data-flow-diagrams)
7. [How to Run](#7-how-to-run)
8. [Dependencies Added](#8-dependencies-added)

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

## 3. Backend — File-by-File

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
    hashed_password  TEXT NOT NULL
);

CREATE TABLE ratings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER NOT NULL,
    movie_id  INTEGER NOT NULL,   -- dense model index (0..9741)
    rating    REAL NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE(user_id, movie_id)     -- re-rating a movie replaces the old value
);
```

**Public functions:**

| Function | Returns | Used for |
|---|---|---|
| `init_db(path)` | — | Called once at startup |
| `execute(sql, params)` | `int` (last row ID) | INSERT, UPDATE, DELETE |
| `fetchone(sql, params)` | `dict \| None` | SELECT one row |
| `fetchall(sql, params)` | `list[dict]` | SELECT multiple rows |

---

### `schemas/user_schema.py` (modified)

**Before:** Empty.

Two Pydantic models for the auth API:

- `AuthRequest` — request body for both register and login: `{username: str, password: str}`
- `TokenResponse` — what the server returns: `{token: str, user_id: int, username: str}`

Pydantic automatically validates incoming JSON and returns HTTP 422 if required fields are missing or have the wrong type.

---

### `schemas/ai_schema.py`

**Location:** `backend/app/schemas/ai_schema.py` (new file)

Four Pydantic models for the AI endpoints:

```python
class RatingRequest:
    movie_id: int    # dense model index shown in the UI
    rating:   float  # 1.0–5.0

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
3. Inserts into the `users` table
4. Returns `{id, username}`

**`login(username, password)`:**
1. Looks up the user by username
2. Verifies the password hash — raises HTTP 401 if wrong
3. Returns `{id, username}`

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

**Location:** `backend/app/routes/auth_routes.py` (new file)

Two endpoints mounted at `/api/auth`:

```
POST /api/auth/register   body: {username, password}  → {token, user_id, username}
POST /api/auth/login      body: {username, password}  → {token, user_id, username}
```

Both call the corresponding `auth_service` function and then `create_token()` to issue a JWT.

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

## 4. Frontend — File-by-File

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

- `login(token, user_id, username)` → writes to `localStorage`, updates state
- `logout()` → clears `localStorage`, sets user to `null`
- `useAuth()` hook → any component can call this to read the user or trigger login/logout

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

**Location:** `frontend/src/pages/RecommendPage.jsx` (new file)

The most interactive page. Two-column layout (single column on mobile via `@media`):

**Left column — recommendation cards:**

Each card shows:
- Rank number (`#1`, `#2`, …)
- Movie title
- Score bar — gradient fill proportional to predicted score out of 5
- "Why?" button → calls `POST /api/explain` with method `"soft"`, opens the `ExplainModal`

**Right column — Genre Override panel (sticky):**

18 genres, each with a 5-level button group:

| Button | Value sent to API | OVERRIDE\_MAP equivalent |
|---|---|---|
| `−−` | `"stark dämpfen"` | −2.0 |
| `−` | `"leicht dämpfen"` | −1.0 |
| `○` | `"neutral"` | 0.0 (default) |
| `+` | `"leicht verstärken"` | +1.0 |
| `++` | `"stark verstärken"` | +2.0 |

"Apply Overrides" sends only non-neutral genres to `POST /api/recommend` and refreshes the list.
Active override count is shown as a badge. "Reset All" clears all overrides and reloads.

**`ExplainModal` component (inline):**

Shown when the user clicks "Why?" on a recommendation. Displays:
- The soft explanation text from `generate_soft_xai_explanation()` (in the teammate's German if core genres are found, or a fallback message)
- A horizontal bar chart of the top 8 genre contributions — positive (blue) and negative (red) — normalised to the maximum absolute value
- A note explaining what the bars mean

---

## 5. API Endpoint Reference

### Auth — `/api/auth`

| Method | Path | Auth | Request body | Response |
|---|---|---|---|---|
| POST | `/register` | None | `{username, password}` | `{token, user_id, username}` |
| POST | `/login` | None | `{username, password}` | `{token, user_id, username}` |

### Movies — `/api/movies`

| Method | Path | Auth | Response |
|---|---|---|---|
| GET | `/popular` | None | `{movies: [{id, title}]}` (50 items) |

### AI — `/api`

All endpoints below require `Authorization: Bearer <token>`.

| Method | Path | Request body | Response |
|---|---|---|---|
| GET | `/genres` | — | `{genres: [string × 18]}` |
| POST | `/ratings` | `{movie_id: int, rating: float}` | `{status: "ok"}` |
| GET | `/ratings` | — | `{ratings: {movie_id: rating}}` |
| GET | `/profile` | — | `{profile: {genre: float 0–1}}` |
| POST | `/recommend` | `{top_n, overrides?, alpha}` | `{recommendations: [{movie_id, title, score}]}` |
| POST | `/recommend/edited-profile` | `{profile: dict, top_n}` | `{recommendations: [...]}` |
| POST | `/explain` | `{movie_id: int, method: "soft"\|"lime"}` | `{method, text?, contributions: [{genre, value}]}` |
| GET | `/importance` | — | `{importance: {genre: float}}` |

---

## 6. Data Flow Diagrams

### Login

```
User submits form
    → POST /api/auth/login {username, password}
    → auth_service.login() verifies sha256 hash in SQLite
    → create_token(user_id) → JWT
    → stored in localStorage
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

---

## 7. How to Run

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
2. Register a new account
3. Rate at least a few movies (the more, the better the profile)
4. Click "Build My Profile →"
5. View your AI-inferred genre affinities
6. Optionally: toggle "Edit Profile" and drag sliders to manually adjust
7. Click "Get Recommendations →"
8. Click "Why?" on any recommendation to see the XAI explanation
9. Use the override panel to boost or suppress genres and observe how recommendations shift

---

## 8. Dependencies Added

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
