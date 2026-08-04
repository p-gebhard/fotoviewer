from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION, DetectionIndexer
from fotoviewer.services.indexing.index import Index


def make_image(path):
    Image.new("RGB", (100, 100), color="red").save(path)


def make_detector(detections_by_call):
    detector = MagicMock()
    detector.detect.side_effect = detections_by_call
    return detector


def make_processor():
    processor = MagicMock()
    processor.compute_image_features.return_value = np.array([[1.0, 0.0]])
    return processor


def test_index_folder_stores_one_entry_per_detection(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    make_image(images_dir / "a.jpg")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    detector = make_detector([
        [{"label": "dog", "score": 0.9, "bbox": [0, 0, 10, 10]}, {"label": "person", "score": 0.8, "bbox": [10, 10, 20, 20]}],
    ])
    processor = make_processor()

    DetectionIndexer(detector, processor).index_folder(str(images_dir), str(output_dir))

    loaded = Index()
    loaded.load(DETECTIONS_COLLECTION, str(output_dir / "detections.npz"))
    assert list(loaded.files(DETECTIONS_COLLECTION)) == ["a.jpg", "a.jpg"]
    assert loaded.num_features(DETECTIONS_COLLECTION) == 2


def test_index_folder_handles_photos_without_detections(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    make_image(images_dir / "empty.jpg")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    detector = make_detector([[]])
    processor = make_processor()

    DetectionIndexer(detector, processor).index_folder(str(images_dir), str(output_dir))

    loaded = Index()
    loaded.load(DETECTIONS_COLLECTION, str(output_dir / "detections.npz"))
    assert loaded.num_features(DETECTIONS_COLLECTION) == 0


def test_index_folder_skips_unreadable_images_without_failing(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    make_image(images_dir / "good.jpg")
    (images_dir / "corrupt.jpg").write_bytes(b"not a real image")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    detector = make_detector([[{"label": "dog", "score": 0.9, "bbox": [0, 0, 10, 10]}]])
    processor = make_processor()

    DetectionIndexer(detector, processor).index_folder(str(images_dir), str(output_dir))

    loaded = Index()
    loaded.load(DETECTIONS_COLLECTION, str(output_dir / "detections.npz"))
    assert list(loaded.files(DETECTIONS_COLLECTION)) == ["good.jpg"]


def test_index_folder_skips_hidden_folders(tmp_path):
    images_dir = tmp_path / "images"
    hidden_dir = images_dir / ".hidden"
    hidden_dir.mkdir(parents=True)
    make_image(images_dir / "visible.jpg")
    make_image(hidden_dir / "inside_hidden.jpg")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    detector = make_detector([[{"label": "dog", "score": 0.9, "bbox": [0, 0, 10, 10]}]])
    processor = make_processor()

    DetectionIndexer(detector, processor).index_folder(str(images_dir), str(output_dir))

    loaded = Index()
    loaded.load(DETECTIONS_COLLECTION, str(output_dir / "detections.npz"))
    assert list(loaded.files(DETECTIONS_COLLECTION)) == ["visible.jpg"]


def test_index_folder_raises_for_missing_folder(tmp_path):
    with pytest.raises(ValueError):
        DetectionIndexer(make_detector([]), make_processor()).index_folder(str(tmp_path / "does-not-exist"), str(tmp_path))
