from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from fotoviewer.services.object_detector.object_detector import ObjectDetector


def make_detector(monkeypatch, model, processor, device="cpu"):
    monkeypatch.setattr(
        "fotoviewer.services.object_detector.object_detector.RTDetrForObjectDetection.from_pretrained",
        lambda name: model,
    )
    monkeypatch.setattr(
        "fotoviewer.services.object_detector.object_detector.RTDetrImageProcessor.from_pretrained",
        lambda name: processor,
    )
    return ObjectDetector(device)


def test_detect_returns_labeled_detections(monkeypatch):
    model = MagicMock()
    model.to.return_value = model
    model.config.id2label = {17: "cat", 18: "dog"}

    processor_output = MagicMock()
    processor_output.to.return_value = {"pixel_values": torch.zeros(1, 3, 640, 640)}
    processor = MagicMock(return_value=processor_output)
    processor.post_process_object_detection.return_value = [
        {
            "scores": torch.tensor([0.94]),
            "labels": torch.tensor([18]),
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        }
    ]

    detector = make_detector(monkeypatch, model, processor)
    detections = detector.detect(Image.new("RGB", (50, 50)))

    assert len(detections) == 1
    assert detections[0]["label"] == "dog"
    assert detections[0]["score"] == pytest.approx(0.94)
    assert detections[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]


def test_model_and_inputs_are_moved_to_the_requested_device(monkeypatch):
    model = MagicMock()
    model.to.return_value = model
    processor_output = MagicMock()
    processor_output.to.return_value = {"pixel_values": torch.zeros(1, 3, 640, 640)}
    processor = MagicMock(return_value=processor_output)
    processor.post_process_object_detection.return_value = [
        {"scores": torch.tensor([]), "labels": torch.tensor([], dtype=torch.long), "boxes": torch.zeros(0, 4)}
    ]

    detector = make_detector(monkeypatch, model, processor, device="cuda")
    detector.detect(Image.new("RGB", (10, 10)))

    model.to.assert_called_once_with("cuda")
    processor_output.to.assert_called_once_with("cuda")
