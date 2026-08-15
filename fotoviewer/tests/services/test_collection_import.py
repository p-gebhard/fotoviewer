from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from fotoviewer.services import image_files
from fotoviewer.services.collection_import import copy_collection, run_import


def make_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color="red").save(path)


def test_copy_collection_copies_new_files(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    make_image(source / "a.jpg")

    result = copy_collection(str(source), dest)

    assert (dest / "a.jpg").exists()
    assert result == {"copied": 1, "skipped": 0, "total": 1}


def test_copy_collection_preserves_subfolder_structure(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    make_image(source / "Urlaub" / "a.jpg")

    copy_collection(str(source), dest)

    assert (dest / "Urlaub" / "a.jpg").exists()


def test_copy_collection_skips_existing_file_with_same_size(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    make_image(source / "a.jpg")
    make_image(dest / "a.jpg")

    result = copy_collection(str(source), dest)

    assert result == {"copied": 0, "skipped": 1, "total": 1}


def test_copy_collection_recopies_file_with_different_size(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    make_image(source / "a.jpg")
    (dest).mkdir(parents=True)
    (dest / "a.jpg").write_bytes(b"not the same content at all, different size")

    result = copy_collection(str(source), dest)

    assert result == {"copied": 1, "skipped": 0, "total": 1}
    assert (dest / "a.jpg").stat().st_size == (source / "a.jpg").stat().st_size


def test_copy_collection_raises_for_missing_source(tmp_path):
    with pytest.raises(ValueError):
        copy_collection(str(tmp_path / "does-not-exist"), tmp_path / "dest")


def test_run_import_completes_all_phases_and_updates_status(tmp_path):
    source = tmp_path / "source"
    images_path = tmp_path / "library"
    index_dir = tmp_path / "indices"
    images_path.mkdir()
    index_dir.mkdir()
    make_image(source / "a.jpg")

    image_files._cache.clear()

    image_processor = MagicMock()
    image_processor.compute_image_features.return_value = np.array([[1.0, 0.0]])
    object_detector = MagicMock()
    object_detector.detect.return_value = [{"label": "dog", "score": 0.9, "bbox": [0, 0, 2, 2]}]

    status = {"running": True, "phase": "idle", "error": None, "copy_result": None}

    run_import(str(source), images_path, str(images_path), str(index_dir), image_processor, object_detector, status)

    assert status["phase"] == "done"
    assert status["running"] is False
    assert status["error"] is None
    assert status["copy_result"] == {"copied": 1, "skipped": 0, "total": 1}
    assert (index_dir / "image.npz").exists()
    assert (index_dir / "detections.npz").exists()


def test_run_import_records_error_and_clears_running_flag(tmp_path):
    status = {"running": True, "phase": "idle", "error": None, "copy_result": None}

    run_import(
        str(tmp_path / "does-not-exist"),
        tmp_path / "library",
        str(tmp_path / "library"),
        str(tmp_path / "indices"),
        MagicMock(),
        MagicMock(),
        status,
    )

    assert status["phase"] == "error"
    assert status["running"] is False
    assert status["error"] is not None
