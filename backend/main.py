import asyncio
import os

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
    # Matches any localhost/127.0.0.1 port rather than hardcoding 5173/5174 --
    # Vite auto-increments to the next free port (5175, 5176, ...) whenever
    # those are already taken, e.g. by another dev server or a previous run
    # that didn't shut down, and a hardcoded allowlist breaks CORS for
    # anyone whose Vite landed on a different port.
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    db.init_db(settings.db_path)
    ai_service.setup(
        settings.model_save_path,
        settings.movies_csv_path,
        settings.ratings_csv_path,
    )
    if not os.path.exists(settings.model_save_path):
        raise RuntimeError(
            f"No trained model found at '{settings.model_save_path}'. "
            "Run `python train.py` from the backend/ directory to train and "
            "save a checkpoint before starting the server."
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
