from fastapi import APIRouter, Depends

from app.database import db
from app.schemas.sus_schema import SUSRequest
from app.utils.jwt_utils import get_current_user

router = APIRouter()

SUS_QUESTIONS = [
    "I think that I would like to use this system frequently.",
    "I found the system unnecessarily complex.",
    "I thought the system was easy to use.",
    "I think that I would need the support of a technical person to use this system.",
    "I found the various functions in this system were well integrated.",
    "I thought there was too much inconsistency in this system.",
    "I would imagine that most people would learn to use this system very quickly.",
    "I found the system very cumbersome to use.",
    "I felt very confident using the system.",
    "I needed to learn a lot of things before I could get going with this system.",
]


def _compute_sus_score(rows: list) -> float | None:
    """Standard SUS scoring formula; None if fewer than all 10 questions were answered."""
    if len(rows) < 10:
        return None
    total = 0
    for r in rows:
        q, resp = r["question_idx"], r["response"]
        total += (resp - 1) if q % 2 == 0 else (5 - resp)
    return total * 2.5


@router.get("/questions")
async def get_questions():
    return {"questions": SUS_QUESTIONS}


@router.post("/submit")
async def submit_sus(req: SUSRequest, user_id: int = Depends(get_current_user)):
    for idx, response in enumerate(req.responses):
        await db.execute(
            "INSERT INTO sus_responses (user_id, question_idx, response) VALUES (?, ?, ?) "
            "ON CONFLICT (user_id, question_idx) DO UPDATE SET response = excluded.response",
            (user_id, idx, response),
        )
    await db.execute(
        "INSERT INTO demographics (user_id, age_group, degree_job, netflix_experience) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT (user_id) DO UPDATE SET age_group = excluded.age_group, "
        "degree_job = excluded.degree_job, netflix_experience = excluded.netflix_experience",
        (user_id, req.age_group, req.degree_job, req.netflix_experience),
    )

    # Store the computed score directly on the user row too, not just the
    # raw per-question responses -- lets a researcher query SUS scores
    # straight from the database without re-deriving the formula.
    rows = await db.fetchall(
        "SELECT question_idx, response FROM sus_responses WHERE user_id = ? ORDER BY question_idx",
        (user_id,),
    )
    score = _compute_sus_score(rows)
    await db.execute(
        "UPDATE users SET sus_done = 1, sus_score = ? WHERE id = ?", (score, user_id)
    )
    return {"done": True}


@router.get("/results")
async def get_results(user_id: int = Depends(get_current_user)):
    """Returns the SUS score (0–100) for this user."""
    rows = await db.fetchall(
        "SELECT question_idx, response FROM sus_responses "
        "WHERE user_id = ? ORDER BY question_idx",
        (user_id,),
    )
    return {"score": _compute_sus_score(rows)}
