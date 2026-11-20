from datetime import datetime
from pathlib import Path

from fotoviewer.services.identity import identify_embedding
from fotoviewer.services.image_files import ImageNotFoundError
from fotoviewer.services.date_cache import DateCache
from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION
from fotoviewer.services.indexing.index import Index


def _in_folder(file_path: str, folder: str) -> bool:
    parent = str(Path(file_path).parent)
    return parent == folder or parent.startswith(folder + "/")


def _matched_individuals(detections_index: Index, individuals_index: Index, name: str, threshold: float) -> set[str]:
    if not detections_index.exists(DETECTIONS_COLLECTION):
        return set()

    files = detections_index.files(DETECTIONS_COLLECTION)
    features = detections_index.features(DETECTIONS_COLLECTION)

    matched = set()
    for file, embedding in zip(files, features):
        identity, _ = identify_embedding(individuals_index, embedding.reshape(1, -1), threshold)
        if identity == name:
            matched.add(str(file))
    return matched


def _year_of(date_cache: DateCache, images_path: str, file_path: str) -> int | None:
    try:
        date_taken = date_cache.get(images_path, file_path)
    except ImageNotFoundError:
        # Index may reference a file that has since been moved or deleted on disk.
        return None
    if not date_taken:
        return None
    try:
        return datetime.strptime(date_taken, "%Y:%m:%d %H:%M:%S").year
    except ValueError:
        return None


def _in_year_range(
    date_cache: DateCache, images_path: str, file_path: str, year_from: int | None, year_to: int | None
) -> bool:
    year = _year_of(date_cache, images_path, file_path)
    if year is None:
        return False
    if year_from is not None and year < year_from:
        return False
    if year_to is not None and year > year_to:
        return False
    return True


def list_folder_options(image_index: Index) -> list[str]:
    if not image_index.exists("image"):
        return []
    folders = {str(Path(f).parent) for f in image_index.files("image")}
    folders.discard(".")
    return sorted(folders)


def filter_library(
    image_index: Index,
    individuals_index: Index,
    detections_index: Index,
    images_path: str,
    identity_threshold: float,
    folder: str | None = None,
    individual: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    date_cache: DateCache | None = None,
) -> list[str]:
    files = [str(f) for f in image_index.files("image")] if image_index.exists("image") else []

    if folder:
        files = [f for f in files if _in_folder(f, folder)]

    if individual:
        matched = _matched_individuals(detections_index, individuals_index, individual, identity_threshold)
        files = [f for f in files if f in matched]

    if year_from is not None or year_to is not None:
        date_cache = date_cache or DateCache()
        files = [f for f in files if _in_year_range(date_cache, images_path, f, year_from, year_to)]
        date_cache.save()

    return sorted(files)
