def test_index_route_renders_template(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "id=\"text_query\"" in response.text


def test_static_files_are_served(client):
    response = client.get("/static/main.css")

    assert response.status_code == 200


def test_image_detail_page_renders_metadata(client, tmp_path):
    from PIL import Image

    Image.new("RGB", (10, 20)).save(tmp_path / "photo.jpg")

    response = client.get("/image", params={"path": "photo.jpg"})

    assert response.status_code == 200
    assert "photo.jpg" in response.text
    assert "10 × 20" in response.text


def test_image_detail_page_404_for_missing_file(client):
    response = client.get("/image", params={"path": "missing.jpg"})

    assert response.status_code == 404
