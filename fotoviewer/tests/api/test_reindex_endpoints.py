import numpy as np
from PIL import Image


def make_image(path):
    Image.new("RGB", (4, 4), color="red").save(path)


def test_start_reindex_returns_202_and_marks_running(client, fake_services, tmp_path):
    fake_services["image_processor"].compute_image_features.return_value = np.array([[1.0, 0.0]])
    fake_services["object_detector"].detect.return_value = []
    make_image(tmp_path / "a.jpg")

    response = client.post("/api/reindex")

    assert response.status_code == 202
    assert client.app.state.reindex_status["running"] is True


def test_start_reindex_rejects_second_reindex_while_running(client):
    client.app.state.reindex_status["running"] = True

    response = client.post("/api/reindex")

    assert response.status_code == 409


def test_start_reindex_rejects_while_import_running(client):
    client.app.state.import_status["running"] = True

    response = client.post("/api/reindex")

    assert response.status_code == 409


def test_start_import_rejects_while_reindex_running(client):
    client.app.state.reindex_status["running"] = True

    response = client.post("/api/import", json={"source_path": "/tmp"})

    assert response.status_code == 409


def test_reindex_status_reflects_app_state(client):
    client.app.state.reindex_status.update(running=True, phase="indexing_images", error=None)

    response = client.get("/api/reindex/status")

    assert response.status_code == 200
    assert response.json()["phase"] == "indexing_images"
