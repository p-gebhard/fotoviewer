from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FOTOVIEWER_")

    images_path: str = "/home/user/Pictures/personal/"
    index_dir: str = "../indices"
    embedding_model_name: str = "google/siglip2-base-patch16-224"
    detection_model_name: str = "PekingU/rtdetr_r50vd"
    identity_match_threshold: float = 0.75
    device: str | None = None
