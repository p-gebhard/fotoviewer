from PIL import Image


def test_metadata_endpoint_returns_dimensions_and_format(client, tmp_path):
    Image.new("RGB", (10, 20)).save(tmp_path / "photo.jpg")

    response = client.get("/api/images/metadata", params={"path": "photo.jpg"})

    assert response.status_code == 200
    body = response.json()
    assert body["width"] == 10
    assert body["height"] == 20
    assert body["format"] == "JPEG"


def test_metadata_endpoint_404_for_missing_file(client):
    response = client.get("/api/images/metadata", params={"path": "missing.jpg"})

    assert response.status_code == 404


def test_metadata_endpoint_blocks_path_traversal(client):
    response = client.get("/api/images/metadata", params={"path": "../../etc/passwd"})

    assert response.status_code == 404
