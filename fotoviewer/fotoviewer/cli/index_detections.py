import torch

from fotoviewer.core.config import Settings
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.detection_indexer import DetectionIndexer
from fotoviewer.services.object_detector.object_detector import ObjectDetector


def main() -> None:
    settings = Settings()
    device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ip = ImageProcessor(settings.embedding_model_name)
    od = ObjectDetector(device, settings.detection_model_name, settings.detection_threshold)
    indexer = DetectionIndexer(od, ip)
    indexer.index_folder(settings.images_path, settings.index_dir)


if __name__ == "__main__":
    main()
