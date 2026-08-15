import re

import numpy as np
from PIL import Image

from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION


def unit(vec):
    return vec / np.linalg.norm(vec)


def make_image(path):
    Image.new("RGB", (4, 4), color="red").save(path)


def test_delete_duplicates_moves_files_and_updates_indices(client, fake_services, tmp_path):
    make_image(tmp_path / "a.jpg")
    fake_services["image_index"].add("image", "a.jpg", unit(np.array([[1.0, 0.0]])))
    fake_services["detections_index"].add(DETECTIONS_COLLECTION, "a.jpg", unit(np.array([[1.0, 0.0]])))

    response = client.post("/api/duplicates/delete", json={"paths": ["a.jpg"]})

    assert response.status_code == 200
    assert response.json() == {"moved": ["a.jpg"], "failed": []}
    assert not (tmp_path / "a.jpg").exists()
    assert list((tmp_path / ".duplicates").glob("*/a.jpg"))
    assert fake_services["image_index"].num_features("image") == 0
    assert fake_services["detections_index"].num_features(DETECTIONS_COLLECTION) == 0
    assert (tmp_path / "image.npz").exists()
    assert (tmp_path / "detections.npz").exists()


def test_delete_duplicates_reflected_immediately_on_sources_page(client, fake_services, tmp_path):
    make_image(tmp_path / "a.jpg")
    fake_services["image_index"].add("image", "a.jpg", unit(np.array([[1.0, 0.0]])))

    client.post("/api/duplicates/delete", json={"paths": ["a.jpg"]})
    response = client.get("/sources")

    assert response.status_code == 200
    match = re.search(r"Einträge</dt>\s*<dd>(\d+)</dd>", response.text)
    assert match and match.group(1) == "0"


def test_delete_duplicates_reports_missing_file_without_failing_request(client, fake_services):
    response = client.post("/api/duplicates/delete", json={"paths": ["does-not-exist.jpg"]})

    assert response.status_code == 200
    assert response.json() == {"moved": [], "failed": ["does-not-exist.jpg"]}


def test_delete_duplicates_422_for_empty_paths(client):
    response = client.post("/api/duplicates/delete", json={"paths": []})

    assert response.status_code == 422
