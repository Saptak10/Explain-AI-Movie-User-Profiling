from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    # deployment branch: Postgres, not SQLite -- a local SQLite file has no
    # persistent disk on Render's free web service tier and gets wiped on
    # every cold-start restart (see app/database/db.py). Defaults to a
    # local Postgres for anyone running this branch outside the deployed
    # instance; set DATABASE_URL to override.
    database_url: str = "postgresql://localhost/explain_ai_db"

    # deployment branch: this service only ever loads a pre-exported ONNX
    # bundle (see export_onnx.py on main) -- it never trains, so there are
    # no movies_csv_path/ratings_csv_path/train_*/personalize_* settings
    # here the way main's config.py has.
    onnx_dir: str = "vectorstore/onnx"

    # Comma-separated list of allowed frontend origins for CORS. Defaults to
    # the local Vite dev server ports; set FRONTEND_ORIGINS in production to
    # the deployed frontend URL(s), e.g. "https://my-app.vercel.app".
    frontend_origins: str = "http://localhost:5173,http://localhost:5174"

    # If onnx_dir's key file doesn't exist on boot and this is set, a zip
    # bundle (model_standard.onnx(.data), model_interactive.onnx(.data),
    # id_mapping.json, importance_aux.npz) is downloaded and extracted into
    # onnx_dir before loading -- lets a deployed backend start from a
    # pre-converted bundle (e.g. a GitHub Release asset) without needing
    # the full checkpoint or MovieLens CSVs at runtime.
    model_download_url: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}

    @property
    def frontend_origins_list(self) -> list[str]:
        return [o.strip() for o in self.frontend_origins.split(",") if o.strip()]


settings = Settings()
