import numpy as np
from PIL import Image

from fotoviewer.services.identity import INDIVIDUALS_COLLECTION
from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION


def unit(vec):
    return vec / np.linalg.norm(vec)


def test_filter_page_renders_all_photos_without_filters(client, fake_services, tmp_path):
    Image.new("RGB", (5, 5)).save(tmp_path / "a.jpg")
    fake_services["image_index"].add("image", "a.jpg", np.array([[1.0, 0.0]]))

    response = client.get("/filter")

    assert response.status_code == 200
    assert "a.jpg" in response.text


def test_filter_page_filters_by_folder_query_param(client, fake_services, tmp_path):
    (tmp_path / "Urlaub").mkdir()
    Image.new("RGB", (5, 5)).save(tmp_path / "Urlaub" / "a.jpg")
    Image.new("RGB", (5, 5)).save(tmp_path / "b.jpg")
    fake_services["image_index"].add("image", "Urlaub/a.jpg", np.array([[1.0, 0.0]]))
    fake_services["image_index"].add("image", "b.jpg", np.array([[0.0, 1.0]]))

    response = client.get("/filter", params={"folder": "Urlaub"})

    assert "Urlaub/a.jpg" in response.text
    assert "\"b.jpg\"" not in response.text and ">b.jpg<" not in response.text


def test_filter_page_shows_hint_when_detections_not_indexed(client, fake_services):
    fake_services["individuals_index"].add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))

    response = client.get("/filter", params={"individual": "Rex"})

    assert response.status_code == 200
    assert "fotoviewer-index-detections" in response.text


def test_filter_page_filters_by_individual_when_detections_indexed(client, fake_services, tmp_path):
    Image.new("RGB", (5, 5)).save(tmp_path / "a.jpg")
    fake_services["image_index"].add("image", "a.jpg", np.array([[1.0, 0.0]]))
    fake_services["individuals_index"].add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))
    fake_services["detections_index"].add(DETECTIONS_COLLECTION, "a.jpg", unit(np.array([[1.0, 0.0]])))

    response = client.get("/filter", params={"individual": "Rex"})

    assert "a.jpg" in response.text
    assert "fotoviewer-index-detections" not in response.text


def test_filter_page_ignores_blank_year_query_params(client, fake_services, tmp_path):
    Image.new("RGB", (5, 5)).save(tmp_path / "a.jpg")
    fake_services["image_index"].add("image", "a.jpg", np.array([[1.0, 0.0]]))

    response = client.get("/filter", params={"year_from": "", "year_to": ""})

    assert response.status_code == 200
    assert "a.jpg" in response.text
