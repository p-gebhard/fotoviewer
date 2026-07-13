from fastapi import Depends, HTTPException, Request, WebSocket

from fotoviewer.services.image_detection import detect_objects_in_file
from fotoviewer.services.image_files import ImageNotFoundError
from fotoviewer.services.image_metadata import get_image_metadata
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.object_detector.object_detector import ObjectDetector


def get_image_processor(websocket: WebSocket) -> ImageProcessor:
    return websocket.app.state.image_processor


def get_object_detector(websocket: WebSocket) -> ObjectDetector:
    return websocket.app.state.object_detector


def get_image_index(websocket: WebSocket) -> Index:
    return websocket.app.state.image_index


def get_object_detector_from_request(request: Request) -> ObjectDetector:
    return request.app.state.object_detector


def get_image_metadata_or_404(path: str, request: Request) -> dict:
    settings = request.app.state.settings
    try:
        return get_image_metadata(settings.images_path, path)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")


def get_image_detections_or_404(
    path: str,
    request: Request,
    object_detector: ObjectDetector = Depends(get_object_detector_from_request),
) -> dict:
    settings = request.app.state.settings
    try:
        return detect_objects_in_file(object_detector, settings.images_path, path)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
