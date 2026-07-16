import numpy as np
from PIL import Image

from fotoviewer.services.identity import INDIVIDUALS_COLLECTION


def test_create_individual_registers_reference_embedding(client, fake_services, tmp_path):
    Image.new("RGB", (50, 50)).save(tmp_path / "photo.jpg")
    fake_services["image_processor"].compute_image_features.return_value = np.array([[1.0, 0.0]])

    response = client.post(
        "/api/individuals",
        json={"path": "photo.jpg", "bbox": [0, 0, 10, 10], "name": "Rex"},
    )

    assert response.status_code == 204
    assert fake_services["individuals_index"].num_features(INDIVIDUALS_COLLECTION) == 1
    assert (tmp_path / "individuals.npz").exists()


def test_create_individual_404_for_missing_file(client):
    response = client.post(
        "/api/individuals",
        json={"path": "missing.jpg", "bbox": [0, 0, 10, 10], "name": "Rex"},
    )

    assert response.status_code == 404
