from fastapi import APIRouter

from app.schemas.user_schema import AuthRequest, TokenResponse
from app.services.auth_service import login, register
from app.utils.jwt_utils import create_token

router = APIRouter()


@router.post("/register", response_model=TokenResponse)
async def do_register(req: AuthRequest):
    user = await register(req.username, req.password)
    return {"token": create_token(user["id"]), "user_id": user["id"], "username": user["username"], "version": user["version"]}


@router.post("/login", response_model=TokenResponse)
async def do_login(req: AuthRequest):
    user = await login(req.username, req.password)
    return {"token": create_token(user["id"]), "user_id": user["id"], "username": user["username"], "version": user["version"]}
