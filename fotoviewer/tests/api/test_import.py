import numpy as np


def test_import_page_renders(client, fake_services, tmp_path):
    fake_services["image_index"].add("image", "Urlaub/a.jpg", np.array([[1.0, 0.0]]))

    response = client.get("/import")

    assert response.status_code == 200
    assert "Urlaub" in response.text


def test_start_import_returns_202_and_marks_running(client, fake_services, tmp_path):
    fake_services["image_processor"].compute_image_features.return_value = np.array([[1.0, 0.0]])
    fake_services["object_detector"].detect.return_value = []
    source = tmp_path / "source"
    source.mkdir()

    response = client.post("/api/import", json={"source_path": str(source)})

    assert response.status_code == 202
    assert client.app.state.import_status["running"] is True


def test_start_import_rejects_second_import_while_running(client):
    client.app.state.import_status["running"] = True

    response = client.post("/api/import", json={"source_path": "/tmp"})

    assert response.status_code == 409


def test_start_import_404_for_path_traversal_in_new_folder_name(client):
    response = client.post(
        "/api/import", json={"source_path": "/tmp", "new_folder_name": "../outside_secret"}
    )

    assert response.status_code == 404


def test_import_status_reflects_app_state(client):
    client.app.state.import_status.update(running=True, phase="copying", error=None, copy_result=None)

    response = client.get("/api/import/status")

    assert response.status_code == 200
    assert response.json()["phase"] == "copying"
