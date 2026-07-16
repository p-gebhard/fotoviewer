from pydantic import BaseModel


class Detection(BaseModel):
    label: str
    score: float
    bbox: list[float]
    identity: str | None = None
    identity_score: float | None = None


class ImageDetections(BaseModel):
    detections: list[Detection]
