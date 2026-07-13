from pydantic import BaseModel


class Detection(BaseModel):
    label: str
    score: float
    bbox: list[float]


class ImageDetections(BaseModel):
    detections: list[Detection]
