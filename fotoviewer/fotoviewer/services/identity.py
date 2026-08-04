from PIL import Image

from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index

INDIVIDUALS_COLLECTION = "individuals"


def crop_detection(image: Image.Image, bbox: list[float]) -> Image.Image:
    return image.crop(tuple(bbox))


def identify_crop(
    image_processor: ImageProcessor, individuals_index: Index, crop_image: Image.Image, threshold: float
) -> tuple[str | None, float | None]:
    num_entries = individuals_index.num_features(INDIVIDUALS_COLLECTION) if individuals_index.exists(INDIVIDUALS_COLLECTION) else 0
    if num_entries == 0:
        return None, None

    features = image_processor.compute_image_features(crop_image)
    matches = individuals_index.search(INDIVIDUALS_COLLECTION, features, k=num_entries)

    # Averaging across all reference photos of a name (instead of trusting a single
    # closest embedding) smooths out one-off false positives from an unlucky crop.
    scores_by_name: dict[str, list[float]] = {}
    for match in matches:
        scores_by_name.setdefault(str(match["file"]), []).append(float(match["score"]))

    best_name, best_score = None, 0.0
    for name, scores in scores_by_name.items():
        average = sum(scores) / len(scores)
        if average > best_score:
            best_name, best_score = name, average

    if best_name is not None and best_score >= threshold:
        return best_name, best_score
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
