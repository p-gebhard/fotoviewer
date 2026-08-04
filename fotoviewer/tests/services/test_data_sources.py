import numpy as np
from PIL import Image

from fotoviewer.core.config import Settings
from fotoviewer.services.data_sources import get_data_overview
from fotoviewer.services.identity import INDIVIDUALS_COLLECTION
from fotoviewer.services.indexing.index import Index


def make_image(path):
    Image.new("RGB", (4, 4), color="red").save(path)


def test_get_data_overview_reports_counts_and_paths(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    make_image(images_dir / "a.jpg")
    make_image(images_dir / "b.jpg")

    index_dir = tmp_path / "indices"
    index_dir.mkdir()

    image_index = Index()
    image_index.create("image")
    image_index.add("image", "a.jpg", np.array([[1.0, 0.0]]))
    (index_dir / "image.npz").write_bytes(b"fake")

    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", np.array([[1.0, 0.0]]))
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", np.array([[0.0, 1.0]]))

    detections_index = Index()
    detections_index.create("detections")
    detections_index.add("detections", "a.jpg", np.array([[1.0, 0.0]]))

    settings = Settings(images_path=str(images_dir), index_dir=str(index_dir))

    overview = get_data_overview(settings, image_index, individuals_index, detections_index)

    assert overview["images_path"] == str(images_dir)
    assert overview["total_files_on_disk"] == 2
    assert overview["image_index"]["num_entries"] == 1
    assert overview["image_index"]["size_bytes"] == 4
    assert overview["individuals_index"]["num_entries"] == 2
    assert overview["individuals_index"]["num_unique_names"] == 1
    assert overview["detections_index"]["num_entries"] == 1
    assert overview["embedding_model_name"] == settings.embedding_model_name
    assert overview["detection_model_name"] == settings.detection_model_name
    assert overview["duplicate_match_threshold"] == settings.duplicate_match_threshold


def test_get_data_overview_handles_missing_index_files(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    index_dir = tmp_path / "indices"
    index_dir.mkdir()

    settings = Settings(images_path=str(images_dir), index_dir=str(index_dir))

    overview = get_data_overview(settings, Index(), Index(), Index())

    assert overview["image_index"]["num_entries"] == 0
    assert overview["image_index"]["size_bytes"] is None
    assert overview["individuals_index"]["num_unique_names"] == 0
    assert overview["detections_index"]["num_entries"] == 0
