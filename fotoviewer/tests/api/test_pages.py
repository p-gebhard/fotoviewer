def test_index_route_renders_template(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Search by Text" in response.text


def test_static_files_are_served(client):
    response = client.get("/static/main.css")

    assert response.status_code == 200
