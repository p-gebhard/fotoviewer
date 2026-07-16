from unittest.mock import MagicMock

import pytest
from PIL import Image

from fotoviewer.services.image_detection import detect_objects_in_file
from fotoviewer.services.image_files import ImageNotFoundError
from fotoviewer.services.indexing.index import Index


def test_detect_objects_in_file_returns_detections(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "a.jpg")
    object_detector = MagicMock()
    object_detector.detect.return_value = [{"label": "dog", "score": 0.9, "bbox": [1, 2, 3, 4]}]
    image_processor = MagicMock()
    individuals_index = Index()  # empty gallery -> no identity match

    result = detect_objects_in_file(
        object_detector, image_processor, individuals_index, str(tmp_path), "a.jpg", 0.75
    )

    assert result == {
        "detections": [
            {"label": "dog", "score": 0.9, "bbox": [1, 2, 3, 4], "identity": None, "identity_score": None}
        ]
    }
    object_detector.detect.assert_called_once()


def test_detect_objects_in_file_raises_for_missing_file(tmp_path):
    object_detector = MagicMock()
    image_processor = MagicMock()
    individuals_index = Index()

    with pytest.raises(ImageNotFoundError):
        detect_objects_in_file(
            object_detector, image_processor, individuals_index, str(tmp_path), "missing.jpg", 0.75
        )
