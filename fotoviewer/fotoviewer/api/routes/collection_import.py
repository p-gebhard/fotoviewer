import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from fotoviewer.api.dependencies import (
    get_image_processor_from_request,
    get_object_detector_from_request,
    get_settings,
)
from fotoviewer.api.errors import not_found_as_http_404
from fotoviewer.core.config import Settings
from fotoviewer.schemas.import_collection import ImportRequest
from fotoviewer.services.collection_import import run_import
from fotoviewer.services.image_files import resolve_or_create_directory_path
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION
from fotoviewer.services.object_detector.object_detector import ObjectDetector

router = APIRouter()


@router.post("/api/import", status_code=202)
def start_import(
    body: ImportRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    image_processor: ImageProcessor = Depends(get_image_processor_from_request),
    object_detector: ObjectDetector = Depends(get_object_detector_from_request),
) -> dict:
    status = request.app.state.import_status
    if status["running"]:
        raise HTTPException(status_code=409, detail="Import läuft bereits")

    dest_relative = "/".join(part for part in [body.destination_folder, body.new_folder_name] if part)
    with not_found_as_http_404("Invalid destination folder"):
        destination_dir = resolve_or_create_directory_path(settings.images_path, dest_relative)

    status.clear()
    status.update(running=True, phase="starting", error=None, copy_result=None)

    def _run():
        run_import(
            body.source_path,
            destination_dir,
            settings.images_path,
            settings.index_dir,
            image_processor,
            object_detector,
            status,
        )
        request.app.state.image_index.load("image", str(Path(settings.index_dir) / "image.npz"))
        request.app.state.detections_index.load(DETECTIONS_COLLECTION, str(Path(settings.index_dir) / "detections.npz"))

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@router.get("/api/import/status")
def import_status(request: Request) -> dict:
    return request.app.state.import_status
