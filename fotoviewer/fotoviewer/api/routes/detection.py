from fastapi import APIRouter, Depends

from fotoviewer.api.dependencies import get_image_detections_or_404
from fotoviewer.schemas.detection import ImageDetections

router = APIRouter()


@router.get("/api/images/detections", response_model=ImageDetections)
def image_detections(detections: dict = Depends(get_image_detections_or_404)):
    return detections
