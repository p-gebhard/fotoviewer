from fastapi import APIRouter, HTTPException, Request

from fotoviewer.schemas.metadata import ImageMetadata
from fotoviewer.services.image_metadata import ImageNotFoundError, get_image_metadata

router = APIRouter()


@router.get("/api/images/metadata", response_model=ImageMetadata)
def image_metadata(path: str, request: Request):
    settings = request.app.state.settings
    try:
        return get_image_metadata(settings.images_path, path)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
