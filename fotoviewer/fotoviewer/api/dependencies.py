from fastapi import WebSocket

from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.object_detector.object_detector import ObjectDetector


def get_image_processor(websocket: WebSocket) -> ImageProcessor:
    return websocket.app.state.image_processor


def get_object_detector(websocket: WebSocket) -> ObjectDetector:
    return websocket.app.state.object_detector


def get_image_index(websocket: WebSocket) -> Index:
    return websocket.app.state.image_index
