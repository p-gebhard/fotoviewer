from PIL import Image

from fotoviewer.services.identity import identify_detections
from fotoviewer.services.image_files import resolve_image_path
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.object_detector.object_detector import ObjectDetector


def detect_objects_in_file(
    object_detector: ObjectDetector,
    image_processor: ImageProcessor,
    individuals_index: Index,
    images_path: str,
    relative_path: str,
    identity_threshold: float,
) -> dict:
    full_path = resolve_image_path(images_path, relative_path)
    with Image.open(full_path) as image:
        image = image.convert("RGB")
        detections = object_detector.detect(image)
        detections = identify_detections(image_processor, individuals_index, image, detections, identity_threshold)
    return {"detections": detections}
