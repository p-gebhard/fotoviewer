import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from fotoviewer.api.dependencies import get_image_processor_from_request, get_object_detector_from_request, get_settings
from fotoviewer.core.config import Settings
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION
from fotoviewer.services.object_detector.object_detector import ObjectDetector
from fotoviewer.services.reindex import run_reindex

router = APIRouter()


@router.post("/api/reindex", status_code=202)
def start_reindex(
    request: Request,
    settings: Settings = Depends(get_settings),
    image_processor: ImageProcessor = Depends(get_image_processor_from_request),
    object_detector: ObjectDetector = Depends(get_object_detector_from_request),
) -> dict:
    status = request.app.state.reindex_status
    if status["running"] or request.app.state.import_status["running"]:
        raise HTTPException(status_code=409, detail="Es läuft bereits ein Indexierungsvorgang")

    status.clear()
    status.update(running=True, phase="starting", error=None)

    def _run():
        run_reindex(settings.images_path, settings.index_dir, image_processor, object_detector, status)
        request.app.state.image_index.load("image", str(Path(settings.index_dir) / "image.npz"))
        request.app.state.detections_index.load(DETECTIONS_COLLECTION, str(Path(settings.index_dir) / "detections.npz"))

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@router.get("/api/reindex/status")
def reindex_status(request: Request) -> dict:
    return request.app.state.reindex_status
