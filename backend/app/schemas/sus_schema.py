from pydantic import BaseModel, field_validator


class SUSRequest(BaseModel):
    responses: list[int]
    age_group: str
    degree_job: str
    netflix_experience: int

    @field_validator("responses")
    @classmethod
    def validate_responses(cls, v):
        if len(v) != 10:
            raise ValueError("Exactly 10 responses required")
        for r in v:
            if r not in range(1, 6):
                raise ValueError("Each response must be 1–5")
        return v

    @field_validator("age_group")
    @classmethod
    def validate_age_group(cls, v):
        valid = {"18-23", "24-30", "30-45", ">45"}
        if v not in valid:
            raise ValueError(f"age_group must be one of {valid}")
        return v

    @field_validator("netflix_experience")
    @classmethod
    def validate_netflix(cls, v):
        if v not in range(1, 6):
            raise ValueError("netflix_experience must be 1–5")
        return v
