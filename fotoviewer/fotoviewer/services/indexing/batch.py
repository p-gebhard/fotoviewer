import json
import os
from typing import Callable

import numpy as np
from PIL import Image
from tqdm import tqdm

from fotoviewer.services.image_files import find_image_paths, invalidate
from fotoviewer.services.indexing.index import Index

MANIFEST_VERSION = 1

ProgressCallback = Callable[[int, int], None]


def manifest_path(index_file: str) -> str:
    return index_file + ".manifest.json"


def _file_stamp(full_path: str) -> list[int]:
    stat = os.stat(full_path)
    return [stat.st_mtime_ns, stat.st_size]


def _load_manifest(path: str, model_key: str) -> dict[str, list[int]] | None:
    """Returns the per-file stamps of the previous run, or None if the previous
    index cannot be reused (missing, unreadable, other format or other model)."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if data.get("version") != MANIFEST_VERSION or data.get("model") != model_key:
        return None
    files = data.get("files")
    return files if isinstance(files, dict) else None


def _write_json_atomic(path: str, data: dict) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def run_batch_indexing(
    images_path: str,
    output_path: str,
    collection: str,
    filename: str,
    desc: str,
    extract: Callable[[Image.Image, str], list[tuple[str, np.ndarray]]],
    model_key: str = "",
    full: bool = False,
    on_progress: ProgressCallback | None = None,
) -> dict:
    """Walks images_path, calls extract(image, rel_path) per photo to get zero or more
    (label, feature) entries, and writes them into a single batch-saved Index collection.

    Incremental by default: a manifest next to the index file records mtime and size of
    every successfully processed photo. Photos whose stamp is unchanged keep their
    existing entries, only new or modified photos are run through extract(), and entries
    of photos that no longer exist are dropped. A different model_key (e.g. after
    switching the embedding model) or full=True rebuilds everything.

    Returns counts: entries (in the saved index), images (on disk), processed (run
    through extract this time), reused (kept from the previous index), removed.
    """
    if not os.path.exists(images_path):
        raise ValueError(f"Folder {images_path} does not exist")

    index_file = os.path.join(output_path, filename)
    manifest_file = manifest_path(index_file)

    invalidate(images_path)  # always index what is on disk now, not a cached walk
    current_stamps = {}
    for full_path in find_image_paths(images_path):
        try:
            current_stamps[os.path.relpath(full_path, images_path)] = _file_stamp(full_path)
        except OSError:
            continue  # vanished between walk and stat

    previous_stamps = None if full or not os.path.exists(index_file) else _load_manifest(manifest_file, model_key)

    labels: list[str] = []
    features: list[np.ndarray] = []
    new_manifest: dict[str, list[int]] = {}
    removed = 0

    if previous_stamps is not None:
        unchanged = {rel for rel, stamp in previous_stamps.items() if current_stamps.get(rel) == stamp}
        removed = sum(1 for rel in previous_stamps if rel not in current_stamps)

        previous = Index()
        previous.load(collection, index_file)
        for label, feature in zip(previous.files(collection), previous.features(collection)):
            if str(label) in unchanged:
                labels.append(str(label))
                features.append(feature.reshape(1, -1))
        for rel in unchanged:
            new_manifest[rel] = current_stamps[rel]

    to_process = [rel for rel in current_stamps if rel not in new_manifest]
    total = len(to_process)
    if on_progress:
        on_progress(0, total)

    for done, rel_path in enumerate(tqdm(to_process, desc=desc, unit="image"), start=1):
        try:
            with Image.open(os.path.join(images_path, rel_path)) as image:
                for label, feature in extract(image, rel_path):
                    labels.append(label)
                    features.append(feature)
            new_manifest[rel_path] = current_stamps[rel_path]
        except Exception as e:
            # Not recorded in the manifest, so it is retried on the next run.
            tqdm.write(f"Ignoring image {rel_path} due to: {e}")
        if on_progress:
            on_progress(done, total)

    index = Index()
    index.create(collection)
    if features:
        index.add_batch(collection, labels, np.vstack(features))
    index.save(collection, index_file)
    _write_json_atomic(manifest_file, {"version": MANIFEST_VERSION, "model": model_key, "files": new_manifest})

    return {
        "entries": len(labels),
        "images": len(current_stamps),
        "processed": total,
        "reused": len(current_stamps) - total,
        "removed": removed,
    }
