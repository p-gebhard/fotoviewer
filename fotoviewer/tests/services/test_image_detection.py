from unittest.mock import MagicMock

import pytest
from PIL import Image

from fotoviewer.services.image_detection import detect_objects_in_file
from fotoviewer.services.image_files import ImageNotFoundError


def test_detect_objects_in_file_returns_detections(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "a.jpg")
    object_detector = MagicMock()
    object_detector.detect.return_value = [{"label": "dog", "score": 0.9, "bbox": [1, 2, 3, 4]}]

    result = detect_objects_in_file(object_detector, str(tmp_path), "a.jpg")

    assert result == {"detections": [{"label": "dog", "score": 0.9, "bbox": [1, 2, 3, 4]}]}
    object_detector.detect.assert_called_once()


def test_detect_objects_in_file_raises_for_missing_file(tmp_path):
    object_detector = MagicMock()

    with pytest.raises(ImageNotFoundError):
        detect_objects_in_file(object_detector, str(tmp_path), "missing.jpg")
