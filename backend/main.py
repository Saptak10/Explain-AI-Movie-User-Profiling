import asyncio
import os
import urllib.request
import zipfile
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


def _download_onnx_bundle_if_configured():
    """Fetch and extract a pre-exported ONNX bundle (zip of
    model_standard.onnx(.data), model_interactive.onnx(.data),
    id_mapping.json, importance_aux.npz -- see export_onnx.py on main) from
    settings.model_download_url when onnx_dir's key file is missing. Lets a
    deployed backend start without needing the full PyTorch checkpoint or
    MovieLens CSVs at runtime."""
    marker = Path(settings.onnx_dir) / "model_standard.onnx"
    if marker.exists() or not settings.model_download_url:
        return
    print(f"Downloading ONNX bundle from {settings.model_download_url} …")
    Path(settings.onnx_dir).mkdir(parents=True, exist_ok=True)
    zip_path = Path(settings.onnx_dir) / "_bundle.zip"
    urllib.request.urlretrieve(settings.model_download_url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(settings.onnx_dir)
    zip_path.unlink()
    print("ONNX bundle downloaded and extracted.")


@app.on_event("startup")
async def startup():
    await db.init_db(settings.database_url)
    ai_service.setup(settings.onnx_dir)
    await asyncio.to_thread(_download_onnx_bundle_if_configured)
    if not (Path(settings.onnx_dir) / "model_standard.onnx").exists():
        raise RuntimeError(
            f"No ONNX bundle found in '{settings.onnx_dir}'. Run "
            "`python export_onnx.py` on the main branch (with a trained "
            "checkpoint) to produce one, or set MODEL_DOWNLOAD_URL to fetch "
            "one automatically."
        )
    print("Loading ONNX model…")
    await asyncio.to_thread(ai_service.load)


app.include_router(auth_router, prefix="/api/auth",   tags=["auth"])
app.include_router(movie_router, prefix="/api/movies", tags=["movies"])
app.include_router(ai_router,   prefix="/api",         tags=["ai"])
app.include_router(sus_router,  prefix="/api/sus",     tags=["sus"])


@app.get("/")
async def health():
    return {"status": "ok"}
