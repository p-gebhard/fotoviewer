from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from fotoviewer.services.identity import (
    INDIVIDUALS_COLLECTION,
    crop_detection,
    identify_crop,
    identify_detections,
    register_individual,
)
from fotoviewer.services.indexing.index import Index


def unit(vec):
    return vec / np.linalg.norm(vec)


def test_crop_detection_crops_to_bbox():
    image = Image.new("RGB", (100, 100))

    cropped = crop_detection(image, [10, 20, 50, 80])

    assert cropped.size == (40, 60)


def test_identify_crop_returns_none_for_empty_gallery():
    image_processor = MagicMock()
    individuals_index = Index()

    identity, score = identify_crop(image_processor, individuals_index, Image.new("RGB", (5, 5)), threshold=0.75)

    assert identity is None
    assert score is None
    image_processor.compute_image_features.assert_not_called()


def test_identify_crop_matches_above_threshold():
    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))

    image_processor = MagicMock()
    image_processor.compute_image_features.return_value = unit(np.array([[1.0, 0.0]]))

    identity, score = identify_crop(image_processor, individuals_index, Image.new("RGB", (5, 5)), threshold=0.75)

    assert identity == "Rex"
    assert score == 1.0


def test_identify_crop_returns_none_below_threshold():
    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))

    image_processor = MagicMock()
    image_processor.compute_image_features.return_value = unit(np.array([[0.0, 1.0]]))  # orthogonal -> score 0.5

    identity, score = identify_crop(image_processor, individuals_index, Image.new("RGB", (5, 5)), threshold=0.75)

    assert identity is None
    assert score is None


def test_identify_detections_enriches_each_detection():
    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))

    image_processor = MagicMock()
    image_processor.compute_image_features.return_value = unit(np.array([[1.0, 0.0]]))

    image = Image.new("RGB", (50, 50))
    detections = [{"label": "dog", "score": 0.9, "bbox": [0, 0, 10, 10]}]

    enriched = identify_detections(image_processor, individuals_index, image, detections, threshold=0.75)

    assert enriched == [
        {"label": "dog", "score": 0.9, "bbox": [0, 0, 10, 10], "identity": "Rex", "identity_score": 1.0}
    ]


def test_register_individual_creates_collection_and_adds_entry():
    individuals_index = Index()
    image_processor = MagicMock()
    image_processor.compute_image_features.return_value = unit(np.array([[1.0, 0.0]]))

    register_individual(image_processor, individuals_index, Image.new("RGB", (50, 50)), [0, 0, 10, 10], "Rex")

    assert individuals_index.num_features(INDIVIDUALS_COLLECTION) == 1
    result = individuals_index.search(INDIVIDUALS_COLLECTION, unit(np.array([[1.0, 0.0]])), k=1)
    assert result[0]["file"] == "Rex"
