import os
from typing import Callable

import numpy as np
from PIL import Image
from tqdm import tqdm

from fotoviewer.services.image_files import find_image_paths
from fotoviewer.services.indexing.index import Index


def run_batch_indexing(
    images_path: str,
    output_path: str,
    collection: str,
    filename: str,
    desc: str,
    extract: Callable[[Image.Image, str], list[tuple[str, np.ndarray]]],
) -> tuple[int, int]:
    """Walks images_path, calls extract(image, rel_path) per photo to get zero or more
    (label, feature) entries, and writes them into a single batch-saved Index collection.
    Returns (num_entries_indexed, num_images_walked).
    """
    if not os.path.exists(images_path):
        raise ValueError(f"Folder {images_path} does not exist")

    image_paths = find_image_paths(images_path)

    labels = []
    features = []
    for full_path in tqdm(image_paths, desc=desc, unit="image"):
        rel_path = os.path.relpath(full_path, images_path)
        try:
            image = Image.open(full_path)
            for label, feature in extract(image, rel_path):
                labels.append(label)
                features.append(feature)
        except Exception as e:
            tqdm.write(f"Ignoring image {rel_path} due to: {e}")

    index = Index()
    index.create(collection)
    if features:
        index.add_batch(collection, labels, np.vstack(features))
    index.save(collection, os.path.join(output_path, filename))

    return len(labels), len(image_paths)
