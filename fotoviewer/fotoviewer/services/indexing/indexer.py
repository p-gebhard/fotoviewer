from typing import Any, Callable

import numpy as np

from fotoviewer.services.indexing.batch import ProgressCallback, run_batch_indexing

IMAGE_COLLECTION = "image"
IMAGE_INDEX_FILENAME = "image.npz"


class Indexer:

    def __init__(self, feature_extractor: Callable[[Any], np.ndarray], model_key: str = "") -> None:
        self._feature_extractor = feature_extractor
        self._model_key = model_key

    def index_folder(
        self, images_path: str, output_path: str, full: bool = False, on_progress: ProgressCallback | None = None
    ) -> dict:
        result = run_batch_indexing(
            images_path,
            output_path,
            IMAGE_COLLECTION,
            IMAGE_INDEX_FILENAME,
            "Indexing images",
            lambda image, rel_path: [(rel_path, self._feature_extractor(image))],
            model_key=self._model_key,
            full=full,
            on_progress=on_progress,
        )
        print(
            f"Image index: {result['entries']} images "
            f"({result['processed']} processed, {result['reused']} unchanged, {result['removed']} removed)"
        )
        return result
