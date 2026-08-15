import os
import shutil
from pathlib import Path

from fotoviewer.services.image_files import find_image_paths, invalidate
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.detection_indexer import DetectionIndexer
from fotoviewer.services.indexing.indexer import Indexer
from fotoviewer.services.object_detector.object_detector import ObjectDetector


def copy_collection(source_path: str, destination_dir: Path) -> dict:
    if not os.path.isdir(source_path):
        raise ValueError(f"Folder {source_path} does not exist")

    source_files = find_image_paths(source_path)

    copied = 0
    skipped = 0
    for full_source in source_files:
        rel = os.path.relpath(full_source, source_path)
        dest = destination_dir / rel
        if dest.exists() and dest.stat().st_size == os.path.getsize(full_source):
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full_source, dest)
        copied += 1

    return {"copied": copied, "skipped": skipped, "total": len(source_files)}


def run_import(
    source_path: str,
    destination_dir: Path,
    images_path: str,
    index_dir: str,
    image_processor: ImageProcessor,
    object_detector: ObjectDetector,
    status: dict,
) -> None:
    try:
        status["phase"] = "copying"
        status["copy_result"] = copy_collection(source_path, destination_dir)

        invalidate(images_path)

        status["phase"] = "indexing_images"
        Indexer(image_processor.compute_image_features).index_folder(images_path, index_dir)

        status["phase"] = "indexing_detections"
        DetectionIndexer(object_detector, image_processor).index_folder(images_path, index_dir)

        status["phase"] = "done"
    except Exception as e:
        status["error"] = str(e)
        status["phase"] = "error"
    finally:
        status["running"] = False
