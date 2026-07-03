from typing import Optional

from pydantic import BaseModel


class AuthRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    token: str
    user_id: int
    username: str
    version: str
    edit_order: Optional[str] = None
