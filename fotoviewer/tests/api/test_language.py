import pytest
from PIL import Image


@pytest.mark.parametrize(
    "language, expected, unexpected",
    [("en", "Search", "Suche"), ("de", "Suche", "Search for a subject")],
)
def test_pages_render_in_configured_language(client, language, expected, unexpected):
    response = client.get("/")

    assert f'<html lang="{language}">' in response.text
    assert expected in response.text
    assert unexpected not in response.text


@pytest.mark.parametrize("language", ["de"])
def test_js_messages_are_embedded_in_configured_language(client, language):
    response = client.get("/sources")

    assert "Baue Such-Index" in response.text


@pytest.mark.parametrize("language", ["en", "de"])
@pytest.mark.parametrize("path", ["/", "/folders", "/sources", "/duplicates", "/individuals", "/filter", "/import"])
def test_all_pages_render_in_all_languages(client, language, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("language, expected", [("en", "Indexing is already in progress"), ("de", "Es läuft bereits")])
def test_conflict_message_is_translated(client, language, expected):
    client.app.state.reindex_status["running"] = True

    response = client.post("/api/reindex")

    assert response.status_code == 409
    assert expected in response.json()["detail"]


@pytest.mark.parametrize("language, expected", [("en", "Duplicate cleanup"), ("de", "Duplikat-Aufräumung")])
def test_duplicate_manifest_is_written_in_configured_language(client, tmp_path, language, expected):
    Image.new("RGB", (4, 4)).save(tmp_path / "a.jpg")

    client.post("/api/duplicates/delete", json={"paths": ["a.jpg"]})

    manifest = next((tmp_path / ".duplicates").glob("*/manifest.txt"))
    assert expected in manifest.read_text()


@pytest.mark.parametrize("language", ["de"])
def test_user_data_in_translated_text_is_escaped(client, fake_services, language, tmp_path):
    import numpy as np
    from fotoviewer.services.identity import INDIVIDUALS_COLLECTION

    fake_services["individuals_index"].add(INDIVIDUALS_COLLECTION, "<script>x</script>", np.array([[1.0, 0.0]]))

    response = client.get("/individuals")

    assert "<script>x</script> ·" not in response.text
    assert "&lt;script&gt;x&lt;/script&gt;" in response.text
