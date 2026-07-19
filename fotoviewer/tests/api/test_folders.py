from PIL import Image


def test_folders_page_lists_subfolders_and_images(client, tmp_path):
    sub_dir = tmp_path / "Urlaub"
    sub_dir.mkdir()
    Image.new("RGB", (4, 4)).save(sub_dir / "beach.jpg")
    Image.new("RGB", (4, 4)).save(tmp_path / "a.jpg")

    response = client.get("/folders")

    assert response.status_code == 200
    assert "Urlaub" in response.text
    assert "a.jpg" in response.text


def test_folders_page_navigates_into_subfolder(client, tmp_path):
    sub_dir = tmp_path / "Urlaub"
    sub_dir.mkdir()
    Image.new("RGB", (4, 4)).save(sub_dir / "beach.jpg")

    response = client.get("/folders", params={"path": "Urlaub"})

    assert response.status_code == 200
    assert "beach.jpg" in response.text


def test_folders_page_404_for_missing_folder(client):
    response = client.get("/folders", params={"path": "does-not-exist"})

    assert response.status_code == 404
