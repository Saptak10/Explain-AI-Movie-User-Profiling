import asyncio
import os
import urllib.request
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.config import settings
from app.database import db
from app.routes.ai_routes import router as ai_router
from app.routes.auth_routes import router as auth_router
from app.routes.movie_routes import router as movie_router
from app.routes.sus_routes import router as sus_router
from app.services.ai_service import ai_service

app = FastAPI(title="Explain-AI Movie Profiling")

app.add_middleware(
    CORSMiddleware,
    # allow_origins covers the deployed frontend (settings.frontend_origins,
    # e.g. the Vercel URL). allow_origin_regex separately matches any local
    # Vite dev server port -- Vite auto-increments past 5173/5174 whenever
    # those are taken, and a hardcoded allowlist would break CORS for
    # anyone whose Vite landed on a different port. Starlette's
    # CORSMiddleware allows an origin if it matches *either*.
    allow_origins=settings.frontend_origins_list,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _download_model_if_configured():
    """Fetch a pre-trained checkpoint from settings.model_download_url when
    the local file is missing -- used in deployments that ship a trained
    model (e.g. a GitHub Release asset) instead of retraining on boot."""
    if os.path.exists(settings.model_save_path) or not settings.model_download_url:
        return
    print(f"Downloading model checkpoint from {settings.model_download_url} …")
    Path(settings.model_save_path).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(settings.model_download_url, settings.model_save_path)
    print("Model checkpoint downloaded.")


@app.on_event("startup")
async def startup():
    db.init_db(settings.db_path)
    ai_service.setup(
        settings.model_save_path,
        settings.movies_csv_path,
        settings.ratings_csv_path,
    )
    await asyncio.to_thread(_download_model_if_configured)
    if not os.path.exists(settings.model_save_path):
        raise RuntimeError(
            f"No trained model found at '{settings.model_save_path}'. "
            "Run `python train.py` from the backend/ directory to train and "
            "save a checkpoint before starting the server, or set "
            "MODEL_DOWNLOAD_URL to fetch one automatically."
        )
    print("Loading saved model…")
    await asyncio.to_thread(ai_service.load)


app.include_router(auth_router, prefix="/api/auth",   tags=["auth"])
app.include_router(movie_router, prefix="/api/movies", tags=["movies"])
app.include_router(ai_router,   prefix="/api",         tags=["ai"])
app.include_router(sus_router,  prefix="/api/sus",     tags=["sus"])


@app.get("/")
async def health():
    return {"status": "ok"}
