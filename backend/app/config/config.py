from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    db_path: str = "app/database/app.db"
    model_save_path: str = "vectorstore/model.pt"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
