from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    db_path: str = "app/database/app.db"
    model_save_path: str = "vectorstore/model.pt"
    movies_csv_path: str = "app/database/ml-latest/movies.csv"
    ratings_csv_path: str = "app/database/ml-latest/ratings.csv"
    train_mask_fraction: float = 0.2
    train_batch_size: int = 32
    train_epochs: int = 25
    train_lr: float = 0.01
    train_lambda_reg: float = 0.05
    train_epsilon_clip: float = 0.15

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
