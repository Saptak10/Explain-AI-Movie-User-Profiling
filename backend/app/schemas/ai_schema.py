from typing import Optional
from pydantic import BaseModel


class RatingRequest(BaseModel):
    movie_id: int
    rating: float


class RecommendRequest(BaseModel):
    top_n: int = 10
    overrides: Optional[dict] = None
    alpha: float = 3.0


class EditedProfileRequest(BaseModel):
    profile: dict
    top_n: int = 10


class ExplainRequest(BaseModel):
    movie_id: int
    method: str = "soft"
