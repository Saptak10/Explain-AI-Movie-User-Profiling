from typing import Optional

from pydantic import BaseModel, Field


class RatingRequest(BaseModel):
    movie_id: int
    rating: float


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
    Manually adjusted genre boost/suppress deltas from the Edit Profile UI
    (e.g. -0.5 for "strongly suppress" through +0.5 for "strongly boost",
    added on top of whatever the AI currently infers for that genre).
    Persisted server-side (see profile_overrides table / ai_routes.py) so
    edits keep affecting /api/profile and /api/recommend on future visits,
    not just this one request. Keys must match real genre names as
    returned by GET /genres -- unknown genre keys are silently ignored by
    the service layer, not rejected here, since the genre vocabulary is
    dynamic and the route should not need to know it in order to validate
    this payload.
    """
    genre_deltas: dict[str, float] = Field(default_factory=dict)
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


class PersonalizedProfileResponse(BaseModel):
    profile: dict[str, float]
    recommendations: list[RecommendedMovie]


class FeatureImportanceItem(BaseModel):
    movie_id: int
    title: str
    importance: float


class ExplanationResponse(BaseModel):
    movie_id: int
    title: str
    rationale: str
    feature_importance: list[FeatureImportanceItem]


class ProfileExplanationResponse(BaseModel):
    profile: dict[str, float]
    genre_explanations: dict[str, list[FeatureImportanceItem]]
