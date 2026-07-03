from typing import Optional

from pydantic import BaseModel, Field


class RatingRequest(BaseModel):
    movie_id: int
    rating: float
    # 0 = initial rating phase, 1 = after round-1 recommendations,
    # 2 = after round-2 recommendations. See profile_edits / ratings.round.
    round: int = 0


class ProfileEditLogRequest(BaseModel):
    round: int
    edit_type: str  # "movie" | "profile"
    genre: str
    level: str
    movie_id: Optional[int] = None


class RecommendLogItem(BaseModel):
    movie_id: int
    position: int
    score: float


class RecommendLogRequest(BaseModel):
    round: int
    rec_type: str  # "initial" | "edited"
    movies: list[RecommendLogItem]


class UserRatingsInput(BaseModel):
    """
    Sparse mapping of movieId (string or int key on the wire; Pydantic
    coerces either) to a float rating. Used wherever a route needs a
    user's full rating history rather than a single rating submission.
    """
    ratings: dict[str, float] = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    top_n: int = 10


class GenreOverrideInput(BaseModel):
    """
    Manually adjusted genre weight sliders from the Interactive Profile UI.
    Keys must match real genre names as returned by GET /genres -- unknown
    genre keys are silently ignored by the service layer (see
    AIService.get_recommendations_from_profile), not rejected here, since
    the genre vocabulary is dynamic and the route should not need to know
    it in order to validate this payload.
    """
    genre_weights: dict[str, float] = Field(default_factory=dict)
    top_n: int = 10


class EditedProfileRequest(BaseModel):
    profile: dict[str, float]
    top_n: int = 10


class ExplainRequest(BaseModel):
    movie_id: int


class RecommendedMovie(BaseModel):
    movie_id: int
    title: str
    score: float
    rationale: Optional[str] = None


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendedMovie]


class UserProfileResponse(BaseModel):
    profile: dict[str, float]


class FeatureImportanceItem(BaseModel):
    movie_id: int
    title: str
    importance: float


class ExplanationResponse(BaseModel):
    movie_id: int
    title: str
    rationale: str
    feature_importance: list[FeatureImportanceItem]
