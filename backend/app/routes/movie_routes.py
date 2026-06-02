from fastapi import APIRouter

from app.services.ai_service import ai_service

router = APIRouter()


@router.get("/popular")
async def popular():
    return {"movies": ai_service.popular_movies}
