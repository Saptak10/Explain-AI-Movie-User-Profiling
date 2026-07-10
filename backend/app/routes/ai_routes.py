import asyncio
import json

from fastapi import APIRouter, Depends


from app.database import db
from app.schemas.ai_schema import (
    ConditionSwitchRequest,
    ExplainRequest,
    GenreOverrideInput,
    RatingRequest,
    RecommendRequest,
)
from app.services.ai_service import ai_service
from app.utils.jwt_utils import get_current_user

router = APIRouter()


# Matches the frontend's LEVELS constant (ProfilePage.jsx / RecommendPage.jsx).
_LEVEL_LABELS = {
    -0.5:  "strong_decrease",
    -0.25: "slight_decrease",
    0.0:   "neutral",
    0.25:  "slight_increase",
    0.5:   "strong_increase",
}


def _level_label(delta: float) -> str:
    return _LEVEL_LABELS.get(delta, f"custom({delta})")


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


async def _get_condition_round(user_id: int) -> tuple[str, int]:
    row = await db.fetchone(
        "SELECT active_version, current_round FROM users WHERE id = ?", (user_id,)
    )
    return row["active_version"], row["current_round"]


async def _advance_round(user_id: int, current_round: int) -> int:
    new_round = current_round + 1
    await db.execute("UPDATE users SET current_round = ? WHERE id = ?", (new_round, user_id))
    return new_round


async def _log_recommendations(user_id: int, version: str, round_: int, trigger_type: str, recs: dict) -> None:
    rows = []
    for list_type in ("top_rated", "for_you"):
        for rank, rec in enumerate(recs.get(list_type, []), start=1):
            rows.append((
                user_id, version, round_, trigger_type, list_type, rank,
                rec["movie_id"], rec["title"], rec["score"],
            ))
    for row in rows:
        await db.execute(
            "INSERT INTO recommendation_events "
            "(user_id, version, round, trigger_type, list_type, rank, movie_id, title, score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )


@router.get("/genres")
async def get_genres():
    return {"genres": ai_service.genres}


@router.post("/ratings")
async def submit_rating(req: RatingRequest, user_id: int = Depends(get_current_user)):
    await db.execute(
        "INSERT OR REPLACE INTO ratings (user_id, movie_id, rating) VALUES (?, ?, ?)",
        (user_id, req.movie_id, req.rating),
    )
    version, round_ = await _get_condition_round(user_id)
    await db.execute(
        "INSERT INTO rating_events (user_id, version, round, movie_id, rating) VALUES (?, ?, ?, ?, ?)",
        (user_id, version, round_, req.movie_id, req.rating),
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
    recs = await asyncio.to_thread(
        ai_service.get_recommendations, ratings, req.top_n, overrides
    )
    version, round_ = await _get_condition_round(user_id)
    new_round = await _advance_round(user_id, round_)
    await _log_recommendations(user_id, version, new_round, "initial", recs)
    return recs


@router.post("/recommend/edited-profile")
async def recommend_edited(req: GenreOverrideInput, user_id: int = Depends(get_current_user)):
    version, round_ = await _get_condition_round(user_id)

    # delta == 0 ("neutral") clears any existing override instead of storing a no-op row.
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
        await db.execute(
            "INSERT INTO profile_edit_events (user_id, version, round, genre, delta, level, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, version, round_, genre, delta, _level_label(delta), req.source),
        )

    ratings = await _get_ratings(user_id)
    overrides = await _get_overrides(user_id)
    recs = await asyncio.to_thread(
        ai_service.get_recommendations, ratings, req.top_n, overrides
    )
    new_round = await _advance_round(user_id, round_)
    await _log_recommendations(user_id, version, new_round, "edited", recs)
    return recs


@router.get("/profile/overrides")
async def get_overrides(user_id: int = Depends(get_current_user)):
    return {"overrides": await _get_overrides(user_id)}


@router.delete("/profile/overrides")
async def clear_overrides(user_id: int = Depends(get_current_user)):
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
    await db.execute("UPDATE users SET has_edited = 1 WHERE id = ?", (user_id,))
    return {"status": "ok"}


@router.post("/user/set-condition")
async def set_condition(req: ConditionSwitchRequest, user_id: int = Depends(get_current_user)):
    # Used by the dev-only preview toggle. Resets current_round; never touches `version`.
    await db.execute(
        "UPDATE users SET active_version = ?, current_round = 0 WHERE id = ?",
        (req.version, user_id),
    )
    await db.execute(
        "INSERT INTO condition_switches (user_id, version) VALUES (?, ?)",
        (user_id, req.version),
    )
    return {"status": "ok", "active_version": req.version}
