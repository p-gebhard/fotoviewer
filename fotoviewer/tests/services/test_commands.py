import base64
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from fotoviewer.services import commands
from fotoviewer.services.indexing.index import Index


class FakeIndex:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, name, embedding, k):
        self.calls.append((name, embedding, k))
        return self._results

    def num_features(self, name):
        return len(self._results)


def _encoded_test_image():
    buf = BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_search_similar_images_searches_the_image_index():
    image_processor = MagicMock()
    image_processor.compute_image_features.return_value = np.array([[1.0, 0.0]])
    image_index = FakeIndex([{"file": "a.jpg", "score": 0.9}])

    result = commands.search_similar_images(image_processor, image_index, _encoded_test_image())

    assert result == [{"file": "a.jpg", "score": 0.9}]
    assert image_index.calls[0][0] == "image"
    image_processor.compute_image_features.assert_called_once()


def test_search_images_by_text_searches_with_text_features():
    image_processor = MagicMock()
    image_processor.compute_text_features.return_value = np.array([[0.0, 1.0]])
    image_index = FakeIndex([{"file": "b.jpg", "score": 0.7}])

    result = commands.search_images_by_text(image_processor, image_index, "a camel")

    assert result == [{"file": "b.jpg", "score": 0.7}]
    image_processor.compute_text_features.assert_called_once_with("a camel")


def test_search_images_by_text_boosts_matching_folder_names():
    image_index = Index()
    image_index.create("image")
    image_index.add("image", "Urlaub/a.jpg", np.array([[0.6, 0.8]]))  # semantic score 0.8, folder matches
    image_index.add("image", "b.jpg", np.array([[0.7, 0.7141428429]]))  # semantic score 0.85, no folder

    image_processor = MagicMock()
    image_processor.compute_text_features.return_value = np.array([[1.0, 0.0]])

    result = commands.search_images_by_text(image_processor, image_index, "Urlaub")

    assert [r["file"] for r in result] == ["Urlaub/a.jpg", "b.jpg"]
    assert result[0]["score"] == pytest.approx(0.9)
    assert result[1]["score"] == pytest.approx(0.85)


def test_search_images_by_text_caps_boosted_score_at_one():
    image_index = Index()
    image_index.create("image")
    image_index.add("image", "Rex/a.jpg", np.array([[1.0, 0.0]]))

    image_processor = MagicMock()
    image_processor.compute_text_features.return_value = np.array([[1.0, 0.0]])

    result = commands.search_images_by_text(image_processor, image_index, "Rex")

    assert result[0]["score"] == pytest.approx(1.0)


def test_search_images_by_text_ignores_single_character_tokens():
    image_index = Index()
    image_index.create("image")
    image_index.add("image", "R/a.jpg", np.array([[0.6, 0.8]]))

    image_processor = MagicMock()
    image_processor.compute_text_features.return_value = np.array([[1.0, 0.0]])

    result = commands.search_images_by_text(image_processor, image_index, "a R")

    assert result[0]["score"] == pytest.approx(0.8)


def test_detect_objects_in_image_returns_encoded_jpeg_and_detections():
    object_detector = MagicMock()
    object_detector.detect.return_value = [{"label": "dog", "score": 0.9, "bbox": [1, 2, 3, 4]}]
    image_processor = MagicMock()
    individuals_index = Index()  # empty gallery -> no identity match

    result = commands.detect_objects_in_image(
        object_detector, image_processor, individuals_index, _encoded_test_image(), 0.75
    )

    assert result["detections"] == [
        {"label": "dog", "score": 0.9, "bbox": [1, 2, 3, 4], "identity": None, "identity_score": None}
    ]
    decoded = base64.b64decode(result["image"])
    Image.open(BytesIO(decoded)).verify()
