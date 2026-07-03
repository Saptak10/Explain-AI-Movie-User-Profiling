import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.database import db
from app.schemas.ai_schema import (
    ExplainRequest,
    GenreOverrideInput,
    ProfileEditLogRequest,
    RatingRequest,
    RecommendRequest,
)
from app.services.ai_service import ai_service
from app.utils.jwt_utils import get_current_user

router = APIRouter()


async def _get_ratings(user_id: int) -> dict:
    rows = await db.fetchall(
        "SELECT movie_id, rating FROM ratings WHERE user_id = ?", (user_id,)
    )
    return {str(r["movie_id"]): r["rating"] for r in rows}


@router.get("/genres")
async def get_genres():
    return {"genres": ai_service.genres}


@router.post("/ratings")
async def submit_rating(req: RatingRequest, user_id: int = Depends(get_current_user)):
    await db.execute(
        "INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, round) VALUES (?, ?, ?, ?)",
        (user_id, req.movie_id, req.rating, req.round),
    )
    return {"status": "ok"}


@router.get("/ratings")
async def get_ratings(user_id: int = Depends(get_current_user)):
    return {"ratings": await _get_ratings(user_id)}


@router.get("/profile")
async def get_profile(user_id: int = Depends(get_current_user)):
    ratings = await _get_ratings(user_id)
    profile = await asyncio.to_thread(ai_service.get_profile, ratings)
    return {"profile": profile}


@router.post("/recommend")
async def recommend(req: RecommendRequest, user_id: int = Depends(get_current_user)):
    ratings = await _get_ratings(user_id)
    recs = await asyncio.to_thread(ai_service.get_recommendations, ratings, req.top_n)
    return {"recommendations": recs}


@router.post("/recommend/edited-profile")
async def recommend_edited(req: GenreOverrideInput, user_id: int = Depends(get_current_user)):
    ratings = await _get_ratings(user_id)
    recs = await asyncio.to_thread(
        ai_service.get_recommendations_from_profile, req.genre_weights, ratings, req.top_n
    )
    return {"recommendations": recs}


@router.post("/explain")
async def explain(req: ExplainRequest, user_id: int = Depends(get_current_user)):
    ratings = await _get_ratings(user_id)
    try:
        result = await asyncio.to_thread(ai_service.explain_movie, req.movie_id, ratings)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result


@router.get("/importance")
async def importance(_: int = Depends(get_current_user)):
    return {"importance": ai_service.global_importance}


@router.post("/user/mark-edited")
async def mark_edited(user_id: int = Depends(get_current_user)):
    await db.execute("UPDATE users SET has_edited = 1 WHERE id = ?", (user_id,))
    return {"status": "ok"}


@router.post("/profile-edits")
async def log_profile_edit(req: ProfileEditLogRequest, user_id: int = Depends(get_current_user)):
    await db.execute(
        "INSERT INTO profile_edits (user_id, round, edit_type, genre, level, movie_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, req.round, req.edit_type, req.genre, req.level, req.movie_id),
    )
    return {"status": "ok"}
