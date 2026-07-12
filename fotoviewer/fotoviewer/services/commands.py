import base64
from io import BytesIO

from PIL import Image

from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.object_detector.object_detector import ObjectDetector
from fotoviewer.services.object_detector.visualizer import render_bboxes


def detect_objects_in_image(object_detector: ObjectDetector, image_b64: str) -> dict:
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    scores, boxes = object_detector.detect(image)
    render_bboxes(image, boxes)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return {
        "image": base64.b64encode(buffered.getvalue()).decode("utf-8"),
        "bboxes": boxes,
        "scores": scores,
    }


def search_similar_images(image_processor: ImageProcessor, image_index: Index, image_b64: str) -> list[dict]:
    image = Image.open(BytesIO(base64.b64decode(image_b64)))
    features = image_processor.compute_image_features(image)
    return image_index.search("image", features, 100)


def search_images_by_text(image_processor: ImageProcessor, image_index: Index, query: str) -> list[dict]:
    features = image_processor.compute_text_features(query)
    return image_index.search("image", features, 100)
