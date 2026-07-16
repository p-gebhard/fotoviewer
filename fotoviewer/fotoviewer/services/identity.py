from PIL import Image

from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index

INDIVIDUALS_COLLECTION = "individuals"


def crop_detection(image: Image.Image, bbox: list[float]) -> Image.Image:
    return image.crop(tuple(bbox))


def identify_crop(
    image_processor: ImageProcessor, individuals_index: Index, crop_image: Image.Image, threshold: float
) -> tuple[str | None, float | None]:
    if not individuals_index.exists(INDIVIDUALS_COLLECTION) or individuals_index.num_features(INDIVIDUALS_COLLECTION) == 0:
        return None, None

    features = image_processor.compute_image_features(crop_image)
    best = individuals_index.search(INDIVIDUALS_COLLECTION, features, k=1)[0]
    if best["score"] >= threshold:
        return str(best["file"]), float(best["score"])
    return None, None


def identify_detections(
    image_processor: ImageProcessor,
    individuals_index: Index,
    image: Image.Image,
    detections: list[dict],
    threshold: float,
) -> list[dict]:
    enriched = []
    for detection in detections:
        crop_image = crop_detection(image, detection["bbox"])
        identity, identity_score = identify_crop(image_processor, individuals_index, crop_image, threshold)
        enriched.append({**detection, "identity": identity, "identity_score": identity_score})
    return enriched


def register_individual(
    image_processor: ImageProcessor, individuals_index: Index, image: Image.Image, bbox: list[float], name: str
) -> None:
    crop_image = crop_detection(image, bbox)
    features = image_processor.compute_image_features(crop_image)
    if not individuals_index.exists(INDIVIDUALS_COLLECTION):
        individuals_index.create(INDIVIDUALS_COLLECTION)
    individuals_index.add(INDIVIDUALS_COLLECTION, name, features)
