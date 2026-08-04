import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from fotoviewer.services.identity import crop_detection
from fotoviewer.services.image_files import find_image_paths
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.object_detector.object_detector import ObjectDetector

DETECTIONS_COLLECTION = "detections"


class DetectionIndexer:

    def __init__(self, object_detector: ObjectDetector, image_processor: ImageProcessor) -> None:
        self._index = Index()
        self._object_detector = object_detector
        self._image_processor = image_processor

    def index_folder(self, images_path: str, output_path: str) -> None:
        if not os.path.exists(images_path):
            raise ValueError(f"Folder {images_path} does not exist")

        image_paths = find_image_paths(images_path)

        rel_paths = []
        features = []
        for full_path in tqdm(image_paths, desc="Indexing detections", unit="image"):
            rel_path = os.path.relpath(full_path, images_path)
            try:
                image = Image.open(full_path).convert("RGB")
                for detection in self._object_detector.detect(image):
                    crop_image = crop_detection(image, detection["bbox"])
                    rel_paths.append(rel_path)
                    features.append(self._image_processor.compute_image_features(crop_image))
            except Exception as e:
                tqdm.write(f"Ignoring image {rel_path} due to: {e}")

        self._index.create(DETECTIONS_COLLECTION)
        if features:
            self._index.add_batch(DETECTIONS_COLLECTION, rel_paths, np.vstack(features))

        self._index.save(DETECTIONS_COLLECTION, os.path.join(output_path, "detections.npz"))
        print(f"Successfully indexed {len(rel_paths)} detections across {len(image_paths)} images")
