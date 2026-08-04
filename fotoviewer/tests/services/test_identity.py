from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from fotoviewer.services.identity import (
    INDIVIDUALS_COLLECTION,
    crop_detection,
    identify_crop,
    identify_detections,
    identify_embedding,
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


def test_identify_crop_averages_multiple_references_of_the_same_name():
    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))  # score 1.0
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[0.0, 1.0]])))  # score 0.5

    image_processor = MagicMock()
    image_processor.compute_image_features.return_value = unit(np.array([[1.0, 0.0]]))

    identity, score = identify_crop(image_processor, individuals_index, Image.new("RGB", (5, 5)), threshold=0.7)

    assert identity == "Rex"
    assert score == pytest.approx(0.75)


def test_identify_crop_ignores_a_single_lucky_match_diluted_by_other_references():
    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    # Rex: two consistently similar references (score 0.8 each).
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[3.0, 4.0]])))
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[3.0, 4.0]])))
    # Mia: one near-perfect match (score 1.0) but a second, clearly different reference (score 0.0).
    individuals_index.add(INDIVIDUALS_COLLECTION, "Mia", unit(np.array([[1.0, 0.0]])))
    individuals_index.add(INDIVIDUALS_COLLECTION, "Mia", unit(np.array([[-1.0, 0.0]])))

    image_processor = MagicMock()
    image_processor.compute_image_features.return_value = unit(np.array([[1.0, 0.0]]))

    identity, score = identify_crop(image_processor, individuals_index, Image.new("RGB", (5, 5)), threshold=0.6)

    # A naive best-single-match would have picked Mia's 1.0 score entry; averaging
    # per name correctly prefers Rex's more consistent 0.8 average instead.
    assert identity == "Rex"
    assert score == pytest.approx(0.8)


def test_identify_embedding_matches_precomputed_embedding_without_image_processor():
    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))

    identity, score = identify_embedding(individuals_index, unit(np.array([[1.0, 0.0]])), threshold=0.75)

    assert identity == "Rex"
    assert score == 1.0


def test_identify_embedding_returns_none_for_empty_gallery():
    identity, score = identify_embedding(Index(), unit(np.array([[1.0, 0.0]])), threshold=0.75)

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
