from pydantic import BaseModel, field_validator


class SUSRequest(BaseModel):
    responses: list[int]

    @field_validator("responses")
    @classmethod
    def validate_responses(cls, v):
        if len(v) != 10:
            raise ValueError("Exactly 10 responses required")
        for r in v:
            if r not in range(1, 6):
                raise ValueError("Each response must be 1–5")
        return v
