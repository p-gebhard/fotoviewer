import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from fotoviewer.services.identity import identify_detections
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.object_detector.object_detector import ObjectDetector
from fotoviewer.services.object_detector.visualizer import render_bboxes

TEXT_SEARCH_RESULT_LIMIT = 100
FOLDER_MATCH_BONUS = 0.1


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


def _folder_name_boost(query: str, file: str) -> float:
    tokens = [token.lower() for token in query.split() if len(token) >= 2]
    if not tokens:
        return 0.0
    folder_segments = [part.lower() for part in Path(file).parent.parts]
    if any(token in segment for token in tokens for segment in folder_segments):
        return FOLDER_MATCH_BONUS
    return 0.0


def search_images_by_text(image_processor: ImageProcessor, image_index: Index, query: str) -> list[dict]:
    features = image_processor.compute_text_features(query)
    num_entries = image_index.num_features("image")
    results = image_index.search("image", features, num_entries)

    for result in results:
        result["score"] = min(1.0, result["score"] + _folder_name_boost(query, str(result["file"])))

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:TEXT_SEARCH_RESULT_LIMIT]
