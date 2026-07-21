import numpy as np


def unit(vec):
    return vec / np.linalg.norm(vec)


def test_duplicates_page_shows_grouped_images(client, fake_services):
    fake_services["image_index"].add("image", "a.jpg", unit(np.array([[1.0, 0.0]])))
    fake_services["image_index"].add("image", "b.jpg", unit(np.array([[0.999, 0.001]])))
    fake_services["image_index"].add("image", "c.jpg", unit(np.array([[0.0, 1.0]])))

    response = client.get("/duplicates")

    assert response.status_code == 200
    assert "a.jpg" in response.text
    assert "b.jpg" in response.text
    assert "c.jpg" not in response.text


def test_duplicates_page_shows_empty_state_when_no_duplicates(client):
    response = client.get("/duplicates")

    assert response.status_code == 200
    assert "Keine Duplikate gefunden" in response.text
