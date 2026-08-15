from pathlib import Path

from fastapi import APIRouter, Depends, Request

from fotoviewer.api.dependencies import get_detections_index_from_request, get_image_index_from_request, get_settings
from fotoviewer.core.config import Settings
from fotoviewer.schemas.duplicates import DeleteDuplicatesRequest
from fotoviewer.services.duplicate_cleanup import move_to_trash
from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION
from fotoviewer.services.indexing.index import Index

router = APIRouter()


@router.post("/api/duplicates/delete")
def delete_duplicates(
    body: DeleteDuplicatesRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    image_index: Index = Depends(get_image_index_from_request),
    detections_index: Index = Depends(get_detections_index_from_request),
) -> dict:
    result = move_to_trash(settings.images_path, body.paths)

    moved = set(result["moved"])
    if moved:
        image_index.remove_entries("image", moved)
        image_index.save("image", str(Path(settings.index_dir) / "image.npz"))

        detections_index.remove_entries(DETECTIONS_COLLECTION, moved)
        detections_index.save(DETECTIONS_COLLECTION, str(Path(settings.index_dir) / "detections.npz"))

    return result
