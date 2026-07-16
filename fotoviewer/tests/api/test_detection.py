from PIL import Image


def test_detections_endpoint_returns_labeled_detections(client, fake_services, tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "photo.jpg")
    fake_services["object_detector"].detect.return_value = [{"label": "dog", "score": 0.9, "bbox": [1, 2, 3, 4]}]

    response = client.get("/api/images/detections", params={"path": "photo.jpg"})

    assert response.status_code == 200
    assert response.json() == {
        "detections": [
            {"label": "dog", "score": 0.9, "bbox": [1, 2, 3, 4], "identity": None, "identity_score": None}
        ]
    }


def test_detections_endpoint_404_for_missing_file(client):
    response = client.get("/api/images/detections", params={"path": "missing.jpg"})

    assert response.status_code == 404
