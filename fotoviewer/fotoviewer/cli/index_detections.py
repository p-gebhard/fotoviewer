import argparse
import os

import torch

from fotoviewer.core.config import Settings
from fotoviewer.services.date_cache import DATE_CACHE_FILENAME, DateCache
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.detection_indexer import DetectionIndexer
from fotoviewer.services.object_detector.object_detector import ObjectDetector
from fotoviewer.services.reindex import detection_model_key, warm_date_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the object index and date cache for /filter (only new/modified photos)."
    )
    parser.add_argument("--full", action="store_true", help="rebuild the index from scratch")
    args = parser.parse_args()

    settings = Settings()
    device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ip = ImageProcessor(settings.embedding_model_name)
    od = ObjectDetector(device, settings.detection_model_name, settings.detection_threshold)
    indexer = DetectionIndexer(od, ip, detection_model_key(od, ip))
    indexer.index_folder(settings.images_path, settings.index_dir, full=args.full)

    warm_date_cache(settings.images_path, DateCache(os.path.join(settings.index_dir, DATE_CACHE_FILENAME)))


if __name__ == "__main__":
    main()
