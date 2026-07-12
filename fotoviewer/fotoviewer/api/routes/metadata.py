from fastapi import APIRouter, Depends

from fotoviewer.api.dependencies import get_image_metadata_or_404
from fotoviewer.schemas.metadata import ImageMetadata

router = APIRouter()


@router.get("/api/images/metadata", response_model=ImageMetadata)
def image_metadata(metadata: dict = Depends(get_image_metadata_or_404)):
    return metadata
