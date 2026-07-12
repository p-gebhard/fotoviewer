import os
import numpy as np
from typing import Any, Callable
from PIL import Image

from fotoviewer.image_processor.image_processor import ImageProcessor
from fotoviewer.image_processor.index import Index


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

class Indexer:

    def __init__(self, feature_extractor: Callable[[Any], np.ndarray]) -> None:
        self._index = Index()
        self._feature_extractor = feature_extractor

    def index_folder(self, images_path: str, output_path: str) -> None:
        if not os.path.exists(images_path):
            raise ValueError(f"Folder {images_path} does not exist")

        indexed_images = 0
        self._index.create("image")
        unexplored_dirs = [images_path]
        while unexplored_dirs:
            dir = unexplored_dirs.pop()
            for file in os.listdir(dir):
                if file[0] == '.':
                    continue
                full_path = os.path.join(dir, file)
                print(full_path)
                if os.path.isdir(full_path):
                    unexplored_dirs.append(full_path)
                    continue
                if os.path.splitext(file.lower())[-1] not in IMAGE_EXTENSIONS:
                    continue
                rel_path = os.path.relpath(full_path, images_path)
                try:
                    image = Image.open(full_path)
                    img_features = self._feature_extractor(image)
                    self._index.add("image", rel_path, img_features)
                    indexed_images += 1
                except Exception as e:
                    print(f"Ignoring image {rel_path} due to: {e}")
        self._index.save("image", os.path.join(output_path, "image.npz"))
        print(f"Successfully created index with {indexed_images} images")
