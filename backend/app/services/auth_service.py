import hashlib
import random
import secrets

from fastapi import HTTPException

from app.database import db


def _hash(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{h}"


def _verify(password: str, stored: str) -> bool:
    parts = stored.split(":", 1)
    if len(parts) != 2:
        return False
    salt, hashed = parts
    return hashlib.sha256((salt + password).encode()).hexdigest() == hashed


async def register(username: str, password: str) -> dict:
    if await db.fetchone("SELECT id FROM users WHERE username = ?", (username,)):
        raise HTTPException(status_code=400, detail="Username already taken")
    version = random.choice(["O", "N"])
    # Counterbalance order only matters for Version O (transparent/editable
    # condition) — N never edits, so edit_order stays NULL for them.
    edit_order = random.choice(["movie_first", "profile_first"]) if version == "O" else None
    uid = await db.execute(
        "INSERT INTO users (username, hashed_password, version, edit_order) VALUES (?, ?, ?, ?)",
        (username, _hash(password), version, edit_order),
    )
    return {"id": uid, "username": username, "version": version, "edit_order": edit_order}


async def login(username: str, password: str) -> dict:
    user = await db.fetchone("SELECT * FROM users WHERE username = ?", (username,))
    if not user or not _verify(password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "id": user["id"],
        "username": user["username"],
        "version": user["version"],
        "edit_order": user["edit_order"],
    }
