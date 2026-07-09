import asyncio
import json

from fastapi import APIRouter, Depends


from app.database import db
from app.schemas.ai_schema import (
    ExplainRequest,
    GenreOverrideInput,
    ProfileEditLogRequest,
    RatingRequest,
    RecommendLogRequest,
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


async def _get_overrides(user_id: int) -> dict:
    rows = await db.fetchall(
        "SELECT genre, delta FROM profile_overrides WHERE user_id = ?", (user_id,)
    )
    return {r["genre"]: r["delta"] for r in rows}


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
    overrides = await _get_overrides(user_id)
    profile = await asyncio.to_thread(ai_service.get_profile, ratings, overrides)
    # Snapshot the profile a researcher would actually see, so it can be
    # queried straight from the database without re-deriving it through
    # the API/model later.
    await db.execute(
        "INSERT INTO profile_snapshots (user_id, profile_json, updated_at) "
        "VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(user_id) DO UPDATE SET profile_json = excluded.profile_json, "
        "updated_at = excluded.updated_at",
        (user_id, json.dumps(profile)),
    )
    return {"profile": profile}


@router.get("/profile/explain")
async def explain_profile(user_id: int = Depends(get_current_user)):
    ratings = await _get_ratings(user_id)
    overrides = await _get_overrides(user_id)
    result = await asyncio.to_thread(
        ai_service.explain_profile, ratings, top_genres=None, overrides=overrides
    )
    return result


@router.post("/recommend")
async def recommend(req: RecommendRequest, user_id: int = Depends(get_current_user)):
    ratings = await _get_ratings(user_id)
    overrides = await _get_overrides(user_id)
    return await asyncio.to_thread(
        ai_service.get_recommendations, ratings, req.top_n, overrides
    )


@router.post("/recommend/edited-profile")
async def recommend_edited(req: GenreOverrideInput, user_id: int = Depends(get_current_user)):
    # Persist each submitted delta -- this is what makes an edit "stick"
    # for every future /api/profile and /api/recommend call, not just this
    # one response. A delta of exactly 0 (the "neutral" level) clears any
    # existing override for that genre instead of storing a no-op row.
    for genre, delta in req.genre_deltas.items():
        if delta == 0:
            await db.execute(
                "DELETE FROM profile_overrides WHERE user_id = ? AND genre = ?",
                (user_id, genre),
            )
        else:
            await db.execute(
                "INSERT INTO profile_overrides (user_id, genre, delta) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id, genre) DO UPDATE SET delta = excluded.delta",
                (user_id, genre, delta),
            )

    ratings = await _get_ratings(user_id)
    overrides = await _get_overrides(user_id)
    return await asyncio.to_thread(
        ai_service.get_recommendations, ratings, req.top_n, overrides
    )


@router.get("/profile/overrides")
async def get_overrides(user_id: int = Depends(get_current_user)):
    """Raw persisted {genre: delta} overrides, so the UI can show which level is currently active per genre."""
    return {"overrides": await _get_overrides(user_id)}


@router.delete("/profile/overrides")
async def clear_overrides(user_id: int = Depends(get_current_user)):
    """Resets a user's profile to the pure AI-inferred one, discarding all saved edits."""
    await db.execute("DELETE FROM profile_overrides WHERE user_id = ?", (user_id,))
    return {"status": "ok"}


@router.post("/profile/personalize")
async def personalize_profile(req: RecommendRequest, user_id: int = Depends(get_current_user)):
    ratings = await _get_ratings(user_id)
    result = await asyncio.to_thread(
        ai_service.create_personalized_profile, ratings, req.top_n
    )
    return result


@router.post("/explain")
async def explain(req: ExplainRequest, user_id: int = Depends(get_current_user)):
    ratings = await _get_ratings(user_id)
    overrides = await _get_overrides(user_id)
    result = await asyncio.to_thread(ai_service.explain_movie, req.movie_id, ratings, overrides)
    return result


@router.get("/importance")
async def importance(_: int = Depends(get_current_user)):
    return {"importance": ai_service.global_importance}


@router.post("/user/mark-edited")
async def mark_edited(user_id: int = Depends(get_current_user)):
    """Record that the user has completed at least one preference edit."""
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


@router.post("/recommendation-log")
async def log_recommendations(req: RecommendLogRequest, user_id: int = Depends(get_current_user)):
    for item in req.movies:
        await db.execute(
            "INSERT INTO recommendation_sessions "
            "(user_id, round, rec_type, movie_id, position, score) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, req.round, req.rec_type, item.movie_id, item.position, item.score),
        )
    return {"status": "ok"}
