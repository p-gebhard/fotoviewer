from unittest.mock import MagicMock

import numpy as np
import torch

from fotoviewer.services.image_processor import ImageProcessor


class FakeOutput:
    def __init__(self, tensor):
        self.pooler_output = tensor


def make_processor(monkeypatch, model, processor):
    monkeypatch.setattr(
        "fotoviewer.services.image_processor.SiglipModel.from_pretrained",
        lambda name: model,
    )
    monkeypatch.setattr(
        "fotoviewer.services.image_processor.SiglipProcessor.from_pretrained",
        lambda name: processor,
    )
    return ImageProcessor()


def test_compute_image_features_returns_normalized_numpy_array(monkeypatch):
    model = MagicMock()
    model.get_image_features.return_value = FakeOutput(torch.tensor([[3.0, 4.0]]))
    processor = MagicMock(return_value={"pixel_values": torch.zeros(1, 3, 2, 2)})

    ip = make_processor(monkeypatch, model, processor)
    result = ip.compute_image_features(object())

    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)
    model.get_image_features.assert_called_once()


def test_compute_text_features_returns_normalized_numpy_array(monkeypatch):
    model = MagicMock()
    model.get_text_features.return_value = FakeOutput(torch.tensor([[1.0, 0.0, 0.0]]))
    processor = MagicMock(return_value={"input_ids": torch.zeros(1, 3, dtype=torch.long)})

    ip = make_processor(monkeypatch, model, processor)
    result = ip.compute_text_features("a camel")

    assert isinstance(result, np.ndarray)
    assert np.allclose(result, [[1.0, 0.0, 0.0]])
    processor.assert_called_once_with(text="a camel", return_tensors="pt", padding="max_length")
    model.get_text_features.assert_called_once()
