from fotoviewer.services.image_files import invalidate
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.detection_indexer import DetectionIndexer
from fotoviewer.services.indexing.indexer import Indexer
from fotoviewer.services.object_detector.object_detector import ObjectDetector


def run_indexers(
    images_path: str,
    index_dir: str,
    image_processor: ImageProcessor,
    object_detector: ObjectDetector,
    status: dict,
) -> None:
    status["phase"] = "indexing_images"
    Indexer(image_processor.compute_image_features).index_folder(images_path, index_dir)

    status["phase"] = "indexing_detections"
    DetectionIndexer(object_detector, image_processor).index_folder(images_path, index_dir)


def run_reindex(
    images_path: str,
    index_dir: str,
    image_processor: ImageProcessor,
    object_detector: ObjectDetector,
    status: dict,
) -> None:
    try:
        invalidate(images_path)
        run_indexers(images_path, index_dir, image_processor, object_detector, status)
        status["phase"] = "done"
    except Exception as e:
        status["error"] = str(e)
        status["phase"] = "error"
    finally:
        status["running"] = False
