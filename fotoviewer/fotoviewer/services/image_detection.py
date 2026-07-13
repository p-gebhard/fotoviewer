from PIL import Image

from fotoviewer.services.image_files import resolve_image_path
from fotoviewer.services.object_detector.object_detector import ObjectDetector


def detect_objects_in_file(object_detector: ObjectDetector, images_path: str, relative_path: str) -> dict:
    full_path = resolve_image_path(images_path, relative_path)
    with Image.open(full_path) as image:
        detections = object_detector.detect(image)
    return {"detections": detections}
