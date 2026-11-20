import numpy as np

from fotoviewer.services.indexing.index import Index
from fotoviewer.services.reindex import reload_indices


def test_search_page_shows_hint_when_index_is_empty(client):
    response = client.get("/")

    assert "fotoviewer-create-index" in response.text


def test_search_page_has_no_hint_when_index_has_entries(client, fake_services):
    fake_services["image_index"].add("image", "a.jpg", np.array([[1.0, 0.0]]))

    response = client.get("/")

    assert "fotoviewer-create-index" not in response.text


def test_search_page_uses_protocol_aware_websocket_url(client):
    assert 'location.protocol === "https:" ? "wss://" : "ws://"' in client.get("/").text


def test_search_on_empty_index_returns_no_results():
    index = Index()
    index.create("image")

    assert index.search("image", np.array([[1.0, 0.0]]), 10) == []


def test_reload_indices_tolerates_missing_index_files(tmp_path):
    class State:
        image_index = Index()
        detections_index = Index()

    reload_indices(State, str(tmp_path))

    assert State.image_index.num_features("image") == 0
