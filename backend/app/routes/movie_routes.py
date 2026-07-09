from fastapi import APIRouter

from app.services.ai_service import ai_service

router = APIRouter()


@router.get("/popular")
async def popular(exclude: str = ""):
    """
    Random sample of movies to rate. `exclude` is an optional
    comma-separated list of movie IDs already shown to the client (e.g.
    the current batch, when the user clicks "Refresh Suggestions"), so a
    refresh surfaces a genuinely different set instead of repeating movies
    the user has already decided they don't recognize.
    """
    exclude_ids = {int(x) for x in exclude.split(",") if x.strip().isdigit()}
    return {"movies": ai_service.get_popular_sample(exclude_ids)}


@router.get("/search")
async def search(q: str = ""):
    """
    Search movies by (partial) title, so a user can rate a specific movie
    by name instead of only what appears in the popular sample.
    """
    return {"movies": ai_service.search_movies(q)}
