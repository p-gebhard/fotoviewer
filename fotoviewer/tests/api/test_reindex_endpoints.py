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


def test_start_reindex_passes_full_flag(client, monkeypatch):
    import fotoviewer.api.routes.reindex as reindex_route

    calls = []
    monkeypatch.setattr(reindex_route, "run_reindex", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(reindex_route, "reload_indices", lambda *args: None)

    client.post("/api/reindex?full=true")
    client.app.state.reindex_status["running"] = False
    client.post("/api/reindex")

    import time
    for _ in range(50):
        if len(calls) == 2:
            break
        time.sleep(0.01)
    assert [c["full"] for c in calls] == [True, False]
