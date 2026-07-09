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
    personalize_epochs: int = 10
    personalize_lr: float = 0.005

    # Comma-separated list of allowed frontend origins for CORS. Defaults to
    # the local Vite dev server ports; set FRONTEND_ORIGINS in production to
    # the deployed frontend URL(s), e.g. "https://my-app.vercel.app".
    frontend_origins: str = "http://localhost:5173,http://localhost:5174"

    # If model_save_path doesn't exist on boot and this is set, it is
    # downloaded before deciding whether to load or train -- lets a
    # deployed backend start from a pre-trained checkpoint (e.g. a GitHub
    # Release asset) without needing the multi-GB MovieLens CSVs at runtime.
    model_download_url: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def frontend_origins_list(self) -> list[str]:
        return [o.strip() for o in self.frontend_origins.split(",") if o.strip()]


settings = Settings()
