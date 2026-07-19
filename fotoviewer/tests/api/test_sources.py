def test_sources_page_renders_overview(client, fake_services):
    fake_services["image_index"]._num_features = 5

    response = client.get("/sources")

    assert response.status_code == 200
    assert "Indizes" in response.text
    assert "5" in response.text
