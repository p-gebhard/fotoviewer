import numpy as np


def test_sources_page_renders_overview(client, fake_services):
    for i in range(5):
        fake_services["image_index"].add("image", f"{i}.jpg", np.array([[1.0, 0.0]]))

    response = client.get("/sources")

    assert response.status_code == 200
    assert "Indizes" in response.text
    assert "5" in response.text
