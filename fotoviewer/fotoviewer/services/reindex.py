import os
from pathlib import Path

from fotoviewer.services.date_cache import DATE_CACHE_FILENAME, DateCache
from fotoviewer.services.image_files import find_image_paths, invalidate
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION, DETECTIONS_INDEX_FILENAME, DetectionIndexer
from fotoviewer.services.indexing.indexer import IMAGE_COLLECTION, IMAGE_INDEX_FILENAME, Indexer
from fotoviewer.services.object_detector.object_detector import ObjectDetector


def image_model_key(image_processor: ImageProcessor) -> str:
    return str(image_processor.model_name)


def detection_model_key(object_detector: ObjectDetector, image_processor: ImageProcessor) -> str:
    # Detection entries depend on the detector, its score threshold and the crop embedding model.
    return f"{object_detector.model_name}|{object_detector.threshold}|{image_processor.model_name}"


def warm_date_cache(images_path: str, date_cache: DateCache) -> None:
    relative_paths = [os.path.relpath(p, images_path) for p in find_image_paths(images_path)]
    date_cache.prune(set(relative_paths))
    date_cache.warm(images_path, relative_paths)
    date_cache.save()


def run_indexers(
    images_path: str,
    index_dir: str,
    image_processor: ImageProcessor,
    object_detector: ObjectDetector,
    status: dict,
    full: bool = False,
    date_cache: DateCache | None = None,
) -> None:
    def on_progress(done: int, total: int) -> None:
        status["progress"] = {"done": done, "total": total}

    status["phase"] = "indexing_images"
    Indexer(image_processor.compute_image_features, image_model_key(image_processor)).index_folder(
        images_path, index_dir, full=full, on_progress=on_progress
    )

    status["phase"] = "indexing_detections"
    DetectionIndexer(object_detector, image_processor, detection_model_key(object_detector, image_processor)).index_folder(
        images_path, index_dir, full=full, on_progress=on_progress
    )

    status["phase"] = "indexing_dates"
    status.pop("progress", None)
    warm_date_cache(images_path, date_cache or DateCache(os.path.join(index_dir, DATE_CACHE_FILENAME)))


def reload_indices(app_state, index_dir: str) -> None:
    # load_or_create: a failed first run may not have written the files yet.
    app_state.image_index.load_or_create(IMAGE_COLLECTION, str(Path(index_dir) / IMAGE_INDEX_FILENAME))
    app_state.detections_index.load_or_create(DETECTIONS_COLLECTION, str(Path(index_dir) / DETECTIONS_INDEX_FILENAME))


def run_reindex(
    images_path: str,
    index_dir: str,
    image_processor: ImageProcessor,
    object_detector: ObjectDetector,
    status: dict,
    full: bool = False,
    date_cache: DateCache | None = None,
) -> None:
    try:
        invalidate(images_path)
        run_indexers(images_path, index_dir, image_processor, object_detector, status, full=full, date_cache=date_cache)
        status["phase"] = "done"
    except Exception as e:
        status["error"] = str(e)
        status["phase"] = "error"
    finally:
        status["running"] = False
