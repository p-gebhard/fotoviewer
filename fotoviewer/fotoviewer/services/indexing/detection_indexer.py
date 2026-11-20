from fotoviewer.services.identity import crop_detection
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.batch import ProgressCallback, run_batch_indexing
from fotoviewer.services.object_detector.object_detector import ObjectDetector

DETECTIONS_COLLECTION = "detections"
DETECTIONS_INDEX_FILENAME = "detections.npz"


class DetectionIndexer:

    def __init__(self, object_detector: ObjectDetector, image_processor: ImageProcessor, model_key: str = "") -> None:
        self._object_detector = object_detector
        self._image_processor = image_processor
        self._model_key = model_key

    def index_folder(
        self, images_path: str, output_path: str, full: bool = False, on_progress: ProgressCallback | None = None
    ) -> dict:
        def extract(image, rel_path):
            image = image.convert("RGB")
            return [
                (rel_path, self._image_processor.compute_image_features(crop_detection(image, detection["bbox"])))
                for detection in self._object_detector.detect(image)
            ]

        result = run_batch_indexing(
            images_path,
            output_path,
            DETECTIONS_COLLECTION,
            DETECTIONS_INDEX_FILENAME,
            "Indexing detections",
            extract,
            model_key=self._model_key,
            full=full,
            on_progress=on_progress,
        )
        print(
            f"Detection index: {result['entries']} detections across {result['images']} images "
            f"({result['processed']} processed, {result['reused']} unchanged, {result['removed']} removed)"
        )
        return result
