from datetime import datetime
from pathlib import Path

from fotoviewer.core.config import Settings
from fotoviewer.services.identity import INDIVIDUALS_COLLECTION
from fotoviewer.services.image_files import find_image_paths
from fotoviewer.services.indexing.index import Index


def _index_file_stats(index: Index, collection: str, file_path: Path) -> dict:
    stats = {
        "path": str(file_path),
        "num_entries": index.num_features(collection) if index.exists(collection) else 0,
        "size_bytes": None,
        "last_modified": None,
    }
    if file_path.exists():
        file_stat = file_path.stat()
        stats["size_bytes"] = file_stat.st_size
        stats["last_modified"] = datetime.fromtimestamp(file_stat.st_mtime).isoformat(timespec="seconds")
    return stats


def get_data_overview(settings: Settings, image_index: Index, individuals_index: Index) -> dict:
    index_dir = Path(settings.index_dir)

    image_stats = _index_file_stats(image_index, "image", index_dir / "image.npz")

    individuals_stats = _index_file_stats(individuals_index, INDIVIDUALS_COLLECTION, index_dir / "individuals.npz")
    individuals_stats["num_unique_names"] = (
        len(set(individuals_index.files(INDIVIDUALS_COLLECTION))) if individuals_index.exists(INDIVIDUALS_COLLECTION) else 0
    )

    return {
        "images_path": settings.images_path,
        "total_files_on_disk": len(find_image_paths(settings.images_path)),
        "embedding_model_name": settings.embedding_model_name,
        "detection_model_name": settings.detection_model_name,
        "identity_match_threshold": settings.identity_match_threshold,
        "image_index": image_stats,
        "individuals_index": individuals_stats,
    }
