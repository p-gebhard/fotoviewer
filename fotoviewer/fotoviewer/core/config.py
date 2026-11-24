from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The directory containing pyproject.toml and .env. Relative paths in the settings are
# resolved against it, so the app behaves the same regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FOTOVIEWER_")

    images_path: str
    index_dir: str = "../indices"
    embedding_model_name: str = "google/siglip2-base-patch16-224"
    detection_model_name: str = "PekingU/rtdetr_r50vd"
    identity_match_threshold: float = 0.75
    duplicate_match_threshold: float = 0.95
    detection_threshold: float = 0.7
    device: str | None = None
    language: Literal["en", "de"] = "en"
    # Weights of the AI restoration models (downloaded on first use).
    model_dir: str = "~/.cache/fotoviewer/models"

    @field_validator("images_path", "index_dir", "model_dir")
    @classmethod
    def _resolve_relative_to_project(cls, value: str) -> str:
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())
