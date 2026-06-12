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
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    db.init_db(settings.db_path)
    ai_service.setup(settings.model_save_path)
    if os.path.exists(settings.model_save_path):
        print("Loading saved model…")
        await asyncio.to_thread(ai_service.load)
    else:
        print("First run — training model (~1-2 min)…")
        await asyncio.to_thread(ai_service.train_and_save)


app.include_router(auth_router, prefix="/api/auth",   tags=["auth"])
app.include_router(movie_router, prefix="/api/movies", tags=["movies"])
app.include_router(ai_router,   prefix="/api",         tags=["ai"])
app.include_router(sus_router,  prefix="/api/sus",     tags=["sus"])


@app.get("/")
async def health():
    return {"status": "ok"}
