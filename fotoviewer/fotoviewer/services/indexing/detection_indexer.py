from fotoviewer.services.identity import crop_detection
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.batch import run_batch_indexing
from fotoviewer.services.object_detector.object_detector import ObjectDetector

DETECTIONS_COLLECTION = "detections"


class DetectionIndexer:

    def __init__(self, object_detector: ObjectDetector, image_processor: ImageProcessor) -> None:
        self._object_detector = object_detector
        self._image_processor = image_processor

    def index_folder(self, images_path: str, output_path: str) -> None:
        def extract(image, rel_path):
            image = image.convert("RGB")
            return [
                (rel_path, self._image_processor.compute_image_features(crop_detection(image, detection["bbox"])))
                for detection in self._object_detector.detect(image)
            ]

        num_indexed, num_images = run_batch_indexing(
            images_path, output_path, DETECTIONS_COLLECTION, "detections.npz", "Indexing detections", extract
        )
        print(f"Successfully indexed {num_indexed} detections across {num_images} images")
