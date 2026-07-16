from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from PIL import Image

from fotoviewer.api.dependencies import get_image_processor_from_request, get_individuals_index_from_request
from fotoviewer.schemas.identity import RegisterIndividualRequest
from fotoviewer.services.identity import INDIVIDUALS_COLLECTION, register_individual
from fotoviewer.services.image_files import ImageNotFoundError, resolve_image_path
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index

router = APIRouter()


@router.post("/api/individuals", status_code=204)
def create_individual(
    body: RegisterIndividualRequest,
    request: Request,
    image_processor: ImageProcessor = Depends(get_image_processor_from_request),
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    settings = request.app.state.settings
    try:
        full_path = resolve_image_path(settings.images_path, body.path)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")

    with Image.open(full_path) as image:
        register_individual(image_processor, individuals_index, image.convert("RGB"), body.bbox, body.name)

    individuals_index.save(INDIVIDUALS_COLLECTION, str(Path(settings.index_dir) / "individuals.npz"))
