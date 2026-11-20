from PIL import Image


def test_thumbnail_returns_jpeg(client, tmp_path):
    Image.new("RGB", (800, 600)).save(tmp_path / "a.jpg")

    response = client.get("/thumbnails", params={"path": "a.jpg"})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-cache"


def test_thumbnail_404_for_missing_image(client):
    assert client.get("/thumbnails", params={"path": "missing.jpg"}).status_code == 404


def test_thumbnail_404_for_path_traversal(client):
    assert client.get("/thumbnails", params={"path": "../../etc/passwd"}).status_code == 404


def test_thumbnail_415_for_unreadable_image(client, tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"garbage")

    assert client.get("/thumbnails", params={"path": "broken.jpg"}).status_code == 415
