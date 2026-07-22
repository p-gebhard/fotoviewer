import re

import numpy as np
from PIL import Image

from fotoviewer.services.identity import INDIVIDUALS_COLLECTION


def register(client, fake_services, tmp_path, name="Rex", filename="photo.jpg", features=None):
    Image.new("RGB", (50, 50)).save(tmp_path / filename)
    fake_services["image_processor"].compute_image_features.return_value = features or np.array([[1.0, 0.0]])
    return client.post(
        "/api/individuals",
        json={"path": filename, "bbox": [0, 0, 10, 10], "name": name},
    )


def test_create_individual_registers_reference_embedding(client, fake_services, tmp_path):
    response = register(client, fake_services, tmp_path)

    assert response.status_code == 204
    assert fake_services["individuals_index"].num_features(INDIVIDUALS_COLLECTION) == 1
    assert (tmp_path / "individuals.npz").exists()
    assert (tmp_path / "individuals_meta.json").exists()


def test_create_individual_404_for_missing_file(client):
    response = client.post(
        "/api/individuals",
        json={"path": "missing.jpg", "bbox": [0, 0, 10, 10], "name": "Rex"},
    )

    assert response.status_code == 404


def test_create_individual_422_for_degenerate_bbox(client, tmp_path):
    Image.new("RGB", (50, 50)).save(tmp_path / "photo.jpg")

    response = client.post(
        "/api/individuals",
        json={"path": "photo.jpg", "bbox": [10, 10, 5, 5], "name": "Rex"},
    )

    assert response.status_code == 422


def test_create_individual_422_for_empty_name(client, tmp_path):
    Image.new("RGB", (50, 50)).save(tmp_path / "photo.jpg")

    response = client.post(
        "/api/individuals",
        json={"path": "photo.jpg", "bbox": [0, 0, 10, 10], "name": "   "},
    )

    assert response.status_code == 422


def test_rename_individual_entry_422_for_empty_name(client, fake_services, tmp_path):
    register(client, fake_services, tmp_path)
    entry_id = client.get("/individuals").text
    entry_id = re.search(r'data-entry-id="([^"]+)"', entry_id).group(1)

    response = client.patch(f"/api/individuals/{entry_id}", json={"name": ""})

    assert response.status_code == 422


def test_individuals_page_lists_registered_entry(client, fake_services, tmp_path):
    register(client, fake_services, tmp_path)

    response = client.get("/individuals")

    assert response.status_code == 200
    assert "Rex" in response.text


def test_individual_thumbnail_returns_cropped_jpeg(client, fake_services, tmp_path):
    register(client, fake_services, tmp_path)
    entry_id = client.get("/individuals").text
    entry_id = re.search(r'data-entry-id="([^"]+)"', entry_id).group(1)

    response = client.get(f"/api/individuals/{entry_id}/thumbnail")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_individual_thumbnail_404_for_unknown_entry(client):
    response = client.get("/api/individuals/does-not-exist/thumbnail")

    assert response.status_code == 404


def test_rename_individual_entry(client, fake_services, tmp_path):
    register(client, fake_services, tmp_path)
    entry_id = client.get("/individuals").text
    entry_id = re.search(r'data-entry-id="([^"]+)"', entry_id).group(1)

    response = client.patch(f"/api/individuals/{entry_id}", json={"name": "Rex (korrigiert)"})

    assert response.status_code == 204
    assert "Rex (korrigiert)" in client.get("/individuals").text


def test_rename_individual_entry_404_for_unknown_id(client):
    response = client.patch("/api/individuals/does-not-exist", json={"name": "New"})

    assert response.status_code == 404


def test_delete_individual_entry(client, fake_services, tmp_path):
    register(client, fake_services, tmp_path)
    entry_id = client.get("/individuals").text
    entry_id = re.search(r'data-entry-id="([^"]+)"', entry_id).group(1)

    response = client.delete(f"/api/individuals/{entry_id}")

    assert response.status_code == 204
    assert fake_services["individuals_index"].num_features(INDIVIDUALS_COLLECTION) == 0


def test_delete_individual_entry_404_for_unknown_id(client):
    response = client.delete("/api/individuals/does-not-exist")

    assert response.status_code == 404


def test_rename_individual_group(client, fake_services, tmp_path):
    register(client, fake_services, tmp_path)

    response = client.patch("/api/individuals/groups/Rex", json={"name": "Rex II"})

    assert response.status_code == 204
    assert "Rex II" in client.get("/individuals").text


def test_delete_individual_group(client, fake_services, tmp_path):
    register(client, fake_services, tmp_path)

    response = client.delete("/api/individuals/groups/Rex")

    assert response.status_code == 204
    assert fake_services["individuals_index"].num_features(INDIVIDUALS_COLLECTION) == 0
