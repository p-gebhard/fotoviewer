from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FOTOVIEWER_")

    images_path: str = "/home/user/Pictures/personal/"
    index_dir: str = "../indices"
    embedding_model_name: str = "google/siglip2-base-patch16-224"
    detection_model_name: str = "PekingU/rtdetr_r50vd"
    identity_match_threshold: float = 0.75
    device: str | None = None
