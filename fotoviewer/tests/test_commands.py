import base64
from io import BytesIO

import numpy as np
from PIL import Image


class FakeIndex:
    def __init__(self, results):
        self._results = results
        self.calls = []

    def search(self, name, embedding, k):
        self.calls.append((name, embedding, k))
        return self._results


def _encoded_test_image():
    buf = BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_search_similar_images_searches_the_image_index(commands_module):
    commands_module.ip.compute_image_features.return_value = np.array([[1.0, 0.0]])
    fake_index = FakeIndex([{"file": "a.jpg", "score": 0.9}])
    commands_module.index = fake_index

    result = commands_module.search_similar_images({"image": _encoded_test_image()})

    assert result == [{"file": "a.jpg", "score": 0.9}]
    assert fake_index.calls[0][0] == "image"
    commands_module.ip.compute_image_features.assert_called_once()


def test_search_images_by_text_searches_with_text_features(commands_module):
    commands_module.ip.compute_text_features.return_value = np.array([[0.0, 1.0]])
    fake_index = FakeIndex([{"file": "b.jpg", "score": 0.7}])
    commands_module.index = fake_index

    result = commands_module.search_images_by_text({"query": "a camel"})

    assert result == [{"file": "b.jpg", "score": 0.7}]
    commands_module.ip.compute_text_features.assert_called_once_with("a camel")


def test_detect_objects_in_image_returns_encoded_jpeg_and_boxes(commands_module):
    commands_module.object_detector.detect.return_value = ([0.9], [[1, 2, 3, 4]])

    result = commands_module.detect_objects_in_image({"image": _encoded_test_image()})

    assert result["scores"] == [0.9]
    assert result["bboxes"] == [[1, 2, 3, 4]]
    decoded = base64.b64decode(result["image"])
    Image.open(BytesIO(decoded)).verify()
