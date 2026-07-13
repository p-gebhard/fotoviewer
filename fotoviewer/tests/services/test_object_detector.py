import pytest
import torch
from PIL import Image

from fotoviewer.services.object_detector.object_detector import ObjectDetector


def make_detector():
    # bypass __init__ so no checkpoint file / CUDA device is required
    return ObjectDetector.__new__(ObjectDetector)


def test_rescale_bboxes_converts_center_format_to_pixel_corners():
    detector = make_detector()
    out_bbox = torch.tensor([[0.5, 0.5, 0.2, 0.4]])

    result = detector.rescale_bboxes(out_bbox, (100, 200))

    assert torch.allclose(result, torch.tensor([[40.0, 60.0, 60.0, 140.0]]))


def test_rescale_bboxes_handles_multiple_boxes():
    detector = make_detector()
    out_bbox = torch.tensor([
        [0.5, 0.5, 1.0, 1.0],
        [0.25, 0.25, 0.5, 0.5],
    ])

    result = detector.rescale_bboxes(out_bbox, (10, 10))

    assert result.shape == (2, 4)
    assert torch.allclose(result[0], torch.tensor([0.0, 0.0, 10.0, 10.0]))
    assert torch.allclose(result[1], torch.tensor([0.0, 0.0, 5.0, 5.0]))


def test_detect_filters_predictions_by_confidence_and_assigns_labels():
    detector = make_detector()
    detector._transform = lambda im: torch.zeros(3, 100, 100)

    logits = torch.full((1, 3, 92), -10.0)
    logits[0, 0, 18] = 10.0  # dog
    logits[0, 1, 18] = -10.0  # low confidence -> filtered out
    logits[0, 2, 1] = 10.0  # person
    pred_boxes = torch.tensor([[
        [0.5, 0.5, 0.2, 0.2],
        [0.1, 0.1, 0.1, 0.1],
        [0.5, 0.5, 0.4, 0.4],
    ]])
    detector._model = lambda img: {"pred_logits": logits, "pred_boxes": pred_boxes}

    detections = detector.detect(Image.new("RGB", (50, 50)))

    assert len(detections) == 2
    assert {d["label"] for d in detections} == {"dog", "person"}
    assert all(isinstance(d["score"], float) and d["score"] > 0.7 for d in detections)
    assert all(len(d["bbox"]) == 4 for d in detections)


def test_detect_rejects_oversized_images():
    detector = make_detector()
    detector._transform = lambda im: torch.zeros(3, 1601, 800)

    with pytest.raises(AssertionError):
        detector.detect(Image.new("RGB", (10, 10)))
