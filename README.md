# Explain-AI Movie User Profiling

An explainable AI system that builds user profiles from movie preferences and generates transparent, interpretable recommendations. Built with a FastAPI backend, React + Vite frontend, MongoDB database, and a full AI/ML stack (LangChain, OpenAI, sentence-transformers, FAISS).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Environment Variables](#environment-variables)
- [Running the App](#running-the-app)
- [Development Guide](#development-guide)
- [Next Steps](#next-steps)

---

## Project Structure

```
Explain-AI-Movie-User-Profiling/
│
├── backend/                    # FastAPI Python backend
│   ├── app/
│   │   ├── ai/                 # LangChain chains, RAG pipelines, explainability logic
│   │   ├── config/             # App settings loaded from .env (config.py)
│   │   ├── database/           # MongoDB connection (db.py)
│   │   ├── models/             # MongoDB document models (user_model.py, ...)
│   │   ├── routes/             # API route handlers (user_routes.py, ...)
│   │   ├── schemas/            # Pydantic request/response schemas
│   │   ├── services/           # Business logic (recommendation, profiling, ...)
│   │   └── utils/              # Shared utility helpers
│   ├── notebooks/              # Jupyter notebooks for EDA & ML experiments
│   ├── uploads/                # User-uploaded files (e.g. watch history CSVs)
│   ├── vectorstore/            # Persisted FAISS vector index
│   ├── main.py                 # FastAPI app entry point
│   ├── requirements.txt        # Python dependencies
│   └── .env                    # Secret keys & config (not committed)
│
├── frontend/                   # React + Vite frontend
│   ├── src/
│   │   ├── assets/             # Images, SVGs
│   │   ├── App.jsx             # Root component
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx            # React DOM entry point
│   ├── public/                 # Static assets served as-is
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── .env                    # Frontend env vars (VITE_ prefixed)
│
├── datasets/                   # Raw & processed movie/user datasets
├── ml_models/                  # Saved model artefacts (.pkl, .pt, etc.)
├── tests/                      # Backend test suite
├── docs/                       # Additional setup guides
│   └── frontend_Setup.md
└── .gitignore
```

---

## Prerequisites

Make sure the following are installed on your machine before starting:

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org/) |
| MongoDB | 7+ | [mongodb.com](https://www.mongodb.com/try/download/community) |
| Git | any | [git-scm.com](https://git-scm.com/) |

---

## Backend Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd Explain-AI-Movie-User-Profiling
```

### 2. Create & activate the virtual environment

```bash
cd backend

# Create (skip if venv/ already exists)
python3 -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

Your terminal prompt should now show `(venv)`.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the template and fill in your real values:

```bash
cp .env .env.local   # optional — or just edit .env directly
```

Open `backend/.env` and replace the placeholder values:

```env
# MongoDB — local default or your Atlas connection string
MONGO_URI=mongodb://localhost:27017/movie_profiling

# LLM API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Auth — generate a strong random string, e.g.: openssl rand -hex 32
JWT_SECRET=your-super-secret-jwt-key-change-this
```

> **Never commit `.env` to git.** It is already in `.gitignore`.

---

## Frontend Setup

### 1. Install Node dependencies

```bash
cd frontend
npm install
```

### 2. Configure frontend environment variables

Create `frontend/.env` (already present but empty):

```env
VITE_API_URL=http://localhost:8000
```

> All Vite env vars **must** be prefixed with `VITE_` to be exposed to the browser. Access them in React with `import.meta.env.VITE_API_URL`.

---

## Running the App

Open **two terminals** — one for the backend, one for the frontend.

### Terminal 1 — Backend

```bash
cd backend
source venv/bin/activate        # Windows: venv\Scripts\activate
uvicorn main:app --reload
```

Backend runs at: `http://localhost:8000`  
Interactive API docs: `http://localhost:8000/docs`

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

Frontend runs at: `http://localhost:5173`

### Make sure MongoDB is running

```bash
# macOS (Homebrew)
brew services start mongodb-community

# Linux (systemd)
sudo systemctl start mongod

# Or run directly
mongod --dbpath /data/db
```

---

## Development Guide

### Where to add new code

| Task | File(s) to edit |
|------|----------------|
| New API endpoint | `backend/app/routes/` — add a new `*_routes.py` |
| Business logic | `backend/app/services/` — one service file per domain |
| Database model | `backend/app/models/` — one model file per collection |
| Request/response shape | `backend/app/schemas/` — Pydantic models |
| App settings | `backend/app/config/config.py` — load from `.env` via `pydantic-settings` |
| MongoDB connection | `backend/app/database/db.py` — Motor async client |
| AI chains / RAG | `backend/app/ai/` — LangChain chains, FAISS retriever |
| React pages/views | `frontend/src/pages/` *(create this folder)* |
| Reusable components | `frontend/src/components/` *(create this folder)* |
| API calls from React | `frontend/src/services/` or `frontend/src/api/` |
| Data exploration | `backend/notebooks/` — Jupyter notebooks |

### Suggested folder additions in `frontend/src/`

```
src/
├── components/     # Reusable UI components
├── pages/          # One file per route/view
├── services/       # Axios calls to the backend
├── hooks/          # Custom React hooks
├── context/        # React context / global state
└── utils/          # Helper functions
```

### Wiring a new backend route into FastAPI

In `backend/main.py`, import and include your router:

```python
from app.routes.user_routes import router as user_router
# from app.routes.movie_routes import router as movie_router

app.include_router(user_router, prefix="/api/users", tags=["users"])
```

### Loading env vars in config

`backend/app/config/config.py` should look like:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mongo_uri: str
    openai_api_key: str
    anthropic_api_key: str
    jwt_secret: str

    class Config:
        env_file = ".env"

settings = Settings()
```

Import `settings` anywhere in the backend:

```python
from app.config.config import settings
```

---

## Next Steps

Rough build order — each step builds on the last:

1. **Config & DB** — fill in `config/config.py` and `database/db.py` (Motor client using `settings.mongo_uri`)
2. **User model & schema** — define the MongoDB user document in `models/user_model.py` and its Pydantic schema in `schemas/user_schema.py`
3. **Auth** — implement register/login in a `services/auth_service.py` and wire up `routes/user_routes.py` (JWT via `python-jose`)
4. **Movie data** — place datasets in `datasets/`, write a loading/preprocessing script or notebook in `backend/notebooks/`
5. **Embeddings & vector store** — embed movie descriptions with `sentence-transformers`, store the FAISS index in `backend/vectorstore/`
6. **AI / recommendation service** — build a LangChain RAG chain in `app/ai/` that retrieves similar movies and explains the recommendation
7. **User profiling** — track and update user preference vectors in MongoDB after each interaction
8. **Explainability layer** — surface SHAP values, attention weights, or natural-language justifications alongside each recommendation
9. **Frontend views** — build React pages for login, movie browsing, recommendations, and the explainability dashboard
10. **Tests** — add pytest tests under `tests/` for each service and route

---

*For frontend-specific setup details see [docs/frontend_Setup.md](docs/frontend_Setup.md).*
