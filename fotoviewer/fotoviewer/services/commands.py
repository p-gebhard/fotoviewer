import base64
from io import BytesIO

from PIL import Image

from fotoviewer.services.identity import identify_detections
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.object_detector.object_detector import ObjectDetector
from fotoviewer.services.object_detector.visualizer import render_bboxes


def detect_objects_in_image(
    object_detector: ObjectDetector,
    image_processor: ImageProcessor,
    individuals_index: Index,
    image_b64: str,
    identity_threshold: float,
) -> dict:
    image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
    detections = object_detector.detect(image)
    detections = identify_detections(image_processor, individuals_index, image, detections, identity_threshold)
    render_bboxes(image, detections)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return {
        "image": base64.b64encode(buffered.getvalue()).decode("utf-8"),
        "detections": detections,
    }


def search_similar_images(image_processor: ImageProcessor, image_index: Index, image_b64: str) -> list[dict]:
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    features = image_processor.compute_image_features(image)
    return image_index.search("image", features, 100)


def search_images_by_text(image_processor: ImageProcessor, image_index: Index, query: str) -> list[dict]:
    features = image_processor.compute_text_features(query)
    return image_index.search("image", features, 100)
