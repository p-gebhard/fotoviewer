from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from fotoviewer.services.reindex import run_reindex


def make_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color="red").save(path)


def test_run_reindex_completes_all_phases_and_updates_status(tmp_path):
    images_path = tmp_path / "library"
    index_dir = tmp_path / "indices"
    images_path.mkdir()
    index_dir.mkdir()
    make_image(images_path / "a.jpg")

    image_processor = MagicMock()
    image_processor.compute_image_features.return_value = np.array([[1.0, 0.0]])
    object_detector = MagicMock()
    object_detector.detect.return_value = []

    status = {"running": True, "phase": "idle", "error": None}

    run_reindex(str(images_path), str(index_dir), image_processor, object_detector, status)

    assert status["phase"] == "done"
    assert status["running"] is False
    assert status["error"] is None
    assert (index_dir / "image.npz").exists()
    assert (index_dir / "detections.npz").exists()


def test_run_reindex_records_error_and_clears_running_flag(tmp_path):
    status = {"running": True, "phase": "idle", "error": None}

    run_reindex(
        str(tmp_path / "does-not-exist"),
        str(tmp_path / "indices"),
        MagicMock(),
        MagicMock(),
        status,
    )

    assert status["phase"] == "error"
    assert status["running"] is False
    assert status["error"] is not None


def test_run_reindex_picks_up_files_added_after_a_recent_cache_population(tmp_path):
    images_path = tmp_path / "library"
    index_dir = tmp_path / "indices"
    images_path.mkdir()
    index_dir.mkdir()
    make_image(images_path / "a.jpg")

    from fotoviewer.services.image_files import find_image_paths
    find_image_paths(str(images_path))  # populate the TTL cache with just a.jpg
    make_image(images_path / "b.jpg")  # added after the cache was populated

    image_processor = MagicMock()
    image_processor.compute_image_features.return_value = np.array([[1.0, 0.0]])
    object_detector = MagicMock()
    object_detector.detect.return_value = []

    status = {"running": True, "phase": "idle", "error": None}
    run_reindex(str(images_path), str(index_dir), image_processor, object_detector, status)

    from fotoviewer.services.indexing.index import Index
    loaded = Index()
    loaded.load("image", str(index_dir / "image.npz"))
    assert loaded.num_features("image") == 2
