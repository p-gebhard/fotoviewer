import os
import numpy as np
from typing import Any, Callable
from PIL import Image
from tqdm import tqdm

from fotoviewer.services.indexing.index import Index


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

class Indexer:

    def __init__(self, feature_extractor: Callable[[Any], np.ndarray]) -> None:
        self._index = Index()
        self._feature_extractor = feature_extractor

    def index_folder(self, images_path: str, output_path: str) -> None:
        if not os.path.exists(images_path):
            raise ValueError(f"Folder {images_path} does not exist")

        image_paths = self._find_image_paths(images_path)

        indexed_images = 0
        self._index.create("image")
        for full_path in tqdm(image_paths, desc="Indexing images", unit="image"):
            rel_path = os.path.relpath(full_path, images_path)
            try:
                image = Image.open(full_path)
                img_features = self._feature_extractor(image)
                self._index.add("image", rel_path, img_features)
                indexed_images += 1
            except Exception as e:
                tqdm.write(f"Ignoring image {rel_path} due to: {e}")

        self._index.save("image", os.path.join(output_path, "image.npz"))
        print(f"Successfully created index with {indexed_images} images")

    def _find_image_paths(self, images_path: str) -> list[str]:
        image_paths = []
        for dirpath, dirnames, filenames in os.walk(images_path):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if filename.startswith("."):
                    continue
                if os.path.splitext(filename.lower())[-1] not in IMAGE_EXTENSIONS:
                    continue
                image_paths.append(os.path.join(dirpath, filename))
        return image_paths
