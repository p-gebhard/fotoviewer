from typing import Any, Callable

import numpy as np

from fotoviewer.services.indexing.batch import run_batch_indexing


class Indexer:

    def __init__(self, feature_extractor: Callable[[Any], np.ndarray]) -> None:
        self._feature_extractor = feature_extractor

    def index_folder(self, images_path: str, output_path: str) -> None:
        num_indexed, _ = run_batch_indexing(
            images_path,
            output_path,
            "image",
            "image.npz",
            "Indexing images",
            lambda image, rel_path: [(rel_path, self._feature_extractor(image))],
        )
        print(f"Successfully created index with {num_indexed} images")
