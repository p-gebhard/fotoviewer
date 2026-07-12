from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FOTOVIEWER_")

    images_path: str = "/home/user/Pictures/personal/"
    index_dir: str = "../indices"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    device: str | None = None
