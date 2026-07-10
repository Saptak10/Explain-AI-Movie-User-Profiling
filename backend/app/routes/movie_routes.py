from fastapi import APIRouter, Depends

from app.database import db
from app.services.ai_service import ai_service
from app.utils.jwt_utils import get_current_user

router = APIRouter()


@router.get("/popular")
async def popular(exclude: str = "", user_id: int = Depends(get_current_user)):
    """
    Random sample of movies to rate. `exclude` is an optional
    comma-separated list of movie IDs already shown to the client (e.g.
    the current batch, when the user clicks "Refresh Suggestions"), so a
    refresh surfaces a genuinely different set instead of repeating movies
    the user has already decided they don't recognize.
    """
    exclude_ids = {int(x) for x in exclude.split(",") if x.strip().isdigit()}
    movies = ai_service.get_popular_sample(exclude_ids)

    row = await db.fetchone(
        "SELECT active_version, current_round FROM users WHERE id = ?", (user_id,)
    )
    for m in movies:
        await db.execute(
            "INSERT INTO shown_movies (user_id, version, round, movie_id) VALUES (?, ?, ?, ?)",
            (user_id, row["active_version"], row["current_round"], m["id"]),
        )

    return {"movies": movies}


@router.get("/search")
async def search(q: str = ""):
    """
    Search movies by (partial) title, so a user can rate a specific movie
    by name instead of only what appears in the popular sample.
    """
    return {"movies": ai_service.search_movies(q)}
