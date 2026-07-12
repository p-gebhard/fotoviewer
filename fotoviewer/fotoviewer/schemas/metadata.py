from pydantic import BaseModel


class ImageMetadata(BaseModel):
    path: str
    width: int
    height: int
    format: str | None
    file_size_bytes: int
    date_taken: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
