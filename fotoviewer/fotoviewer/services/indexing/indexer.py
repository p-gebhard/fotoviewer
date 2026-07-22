import os
import numpy as np
from typing import Any, Callable
from PIL import Image
from tqdm import tqdm

from fotoviewer.services.image_files import find_image_paths
from fotoviewer.services.indexing.index import Index


class Indexer:

    def __init__(self, feature_extractor: Callable[[Any], np.ndarray]) -> None:
        self._index = Index()
        self._feature_extractor = feature_extractor

    def index_folder(self, images_path: str, output_path: str) -> None:
        if not os.path.exists(images_path):
            raise ValueError(f"Folder {images_path} does not exist")

        image_paths = find_image_paths(images_path)

        rel_paths = []
        features = []
        for full_path in tqdm(image_paths, desc="Indexing images", unit="image"):
            rel_path = os.path.relpath(full_path, images_path)
            try:
                image = Image.open(full_path)
                img_features = self._feature_extractor(image)
                rel_paths.append(rel_path)
                features.append(img_features)
            except Exception as e:
                tqdm.write(f"Ignoring image {rel_path} due to: {e}")

        self._index.create("image")
        if features:
            self._index.add_batch("image", rel_paths, np.vstack(features))

        self._index.save("image", os.path.join(output_path, "image.npz"))
        print(f"Successfully created index with {len(rel_paths)} images")
