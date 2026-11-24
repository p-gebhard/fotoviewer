from io import BytesIO

import pytest
from PIL import Image


@pytest.fixture
def photo(tmp_path):
    Image.new("RGB", (400, 300), (60, 70, 120)).save(tmp_path / "a.jpg")
    return "a.jpg"


def test_preview_returns_jpeg(client, photo):
    response = client.post("/api/images/enhance/preview", json={"path": photo, "adjustments": {"exposure": 1}})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert Image.open(BytesIO(response.content)).size == (400, 300)


def test_preview_rejects_out_of_range_values(client, photo):
    response = client.post("/api/images/enhance/preview", json={"path": photo, "adjustments": {"exposure": 5}})
    assert response.status_code == 422


@pytest.mark.parametrize("path", ["missing.jpg", "../../etc/passwd"])
def test_preview_404(client, path):
    assert client.post("/api/images/enhance/preview", json={"path": path}).status_code == 404


def test_preview_415_for_unreadable_image(client, tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"garbage")
    assert client.post("/api/images/enhance/preview", json={"path": "broken.jpg"}).status_code == 415


def test_auto_returns_all_slider_values(client, photo):
    response = client.get("/api/images/enhance/auto", params={"path": photo})

    assert response.status_code == 200
    values = response.json()
    assert set(values) == {
        "exposure", "contrast", "highlights", "shadows", "temperature", "tint", "saturation", "vibrance", "sharpen",
    }
    assert values["temperature"] > 0  # bluish test image gets warmed up


def test_save_writes_copy_and_starts_reindex(client, photo, tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(
        "fotoviewer.api.routes.enhance.start_background_reindex", lambda app, settings: started.append(True) or True
    )

    response = client.post("/api/images/enhance/save", json={"path": photo, "adjustments": {"contrast": 0.5}})

    assert response.status_code == 200
    assert response.json() == {"path": "a_edited.jpg", "reindex_started": True}
    assert (tmp_path / "a_edited.jpg").is_file()
    assert started == [True]


def test_save_does_not_start_second_reindex(client, photo, monkeypatch):
    client.app.state.reindex_status["running"] = True

    response = client.post("/api/images/enhance/save", json={"path": photo})

    assert response.status_code == 200
    assert response.json()["reindex_started"] is False


@pytest.mark.parametrize("language", ["en", "de"])
def test_detail_page_contains_edit_panel(client, photo, language):
    html = client.get("/image", params={"path": photo}).text
    assert 'id="edit_panel"' in html
    assert ("Save as copy" if language == "en" else "Als Kopie speichern") in html


def _wait_for_restoration(client, timeout=10):
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/images/restore/status").json()
        if not status["running"]:
            return status
        time.sleep(0.05)
    raise AssertionError("restoration did not finish")


def test_restore_creates_working_image_used_by_preview_and_save(client, photo, tmp_path):
    response = client.post("/api/images/restore", json={"path": photo, "tool": "denoise"})
    assert response.status_code == 202

    status = _wait_for_restoration(client)
    assert status["phase"] == "done"
    working = status["result"]
    assert working["steps"] == ["denoise"]

    # The fake model inverts the image, so the preview of the working image differs from the original.
    preview = client.post("/api/images/enhance/preview", json={"path": photo, "working_id": working["id"]})
    assert preview.status_code == 200
    assert Image.open(BytesIO(preview.content)).getpixel((10, 10))[0] > 150

    saved = client.post("/api/images/enhance/save", json={"path": photo, "working_id": working["id"]})
    assert saved.status_code == 200
    with Image.open(tmp_path / saved.json()["path"]) as copy:
        assert copy.getpixel((10, 10))[0] > 150  # original red is 60, inverted 195
    with Image.open(tmp_path / photo) as original:
        assert original.getpixel((10, 10))[0] < 100


def test_restore_steps_can_be_chained(client, photo):
    client.post("/api/images/restore", json={"path": photo, "tool": "retouch"})
    first = _wait_for_restoration(client)["result"]

    client.post("/api/images/restore", json={"path": photo, "tool": "upscale_2x", "working_id": first["id"]})
    second = _wait_for_restoration(client)["result"]

    assert second["steps"] == ["retouch", "upscale_2x"]
    assert (second["width"], second["height"]) == (800, 600)


def test_restore_rejects_unknown_tool(client, photo):
    assert client.post("/api/images/restore", json={"path": photo, "tool": "magic"}).status_code == 422


def test_restore_rejects_too_large_output(client, tmp_path):
    Image.new("RGB", (6000, 5000)).save(tmp_path / "big.png")
    response = client.post("/api/images/restore", json={"path": "big.png", "tool": "upscale_4x"})
    assert response.status_code == 422
    assert "100" in response.json()["detail"]


@pytest.mark.parametrize("body", [{"path": "missing.jpg"}, {"path": "a.jpg", "working_id": "0" * 32}])
def test_restore_404(client, photo, body):
    assert client.post("/api/images/restore", json={**body, "tool": "denoise"}).status_code == 404


def test_restore_409_while_running(client, photo):
    client.app.state.restore_status["running"] = True
    assert client.post("/api/images/restore", json={"path": photo, "tool": "denoise"}).status_code == 409


def test_preview_404_for_working_image_of_other_photo(client, photo, tmp_path):
    Image.new("RGB", (40, 30)).save(tmp_path / "b.jpg")
    client.post("/api/images/restore", json={"path": photo, "tool": "denoise"})
    working = _wait_for_restoration(client)["result"]

    response = client.post("/api/images/enhance/preview", json={"path": "b.jpg", "working_id": working["id"]})
    assert response.status_code == 404


def test_detail_page_lists_ai_tools_with_download_hint(client, photo):
    html = client.get("/image", params={"path": photo}).text
    assert 'data-tool="denoise"' in html
    assert "downloads 72 MB" in html


def _mask_data_url(size=(400, 300), box=(150, 100, 250, 200)):
    import base64

    mask = Image.new("RGBA", size, (0, 0, 0, 0))
    mask.paste((255, 0, 0, 255), box)
    buffer = BytesIO()
    mask.save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_inpaint_with_painted_mask(client, photo, tmp_path):
    response = client.post("/api/images/restore", json={"path": photo, "tool": "inpaint", "mask": _mask_data_url()})
    assert response.status_code == 202

    status = _wait_for_restoration(client)
    assert status["phase"] == "done", status
    assert status["result"]["steps"] == ["inpaint"]


@pytest.mark.parametrize("mask", [None, "data:image/png;base64,!!!", _mask_data_url(box=(0, 0, 0, 0))])
def test_inpaint_rejects_missing_invalid_or_empty_mask(client, photo, mask):
    response = client.post("/api/images/restore", json={"path": photo, "tool": "inpaint", "mask": mask})
    assert response.status_code == 422


def test_faces_with_strength(client, photo):
    response = client.post("/api/images/restore", json={"path": photo, "tool": "faces", "strength": 0.5})
    assert response.status_code == 202
    assert _wait_for_restoration(client)["phase"] == "done"


def test_faces_reports_when_none_found(client, tmp_path):
    Image.new("RGB", (100, 80)).save(tmp_path / "small.jpg")  # fake detector finds no face in small images
    client.post("/api/images/restore", json={"path": "small.jpg", "tool": "faces"})
    status = _wait_for_restoration(client)
    assert status["phase"] == "error" and status["error_key"] == "no_faces"


def test_strength_out_of_range_rejected(client, photo):
    response = client.post("/api/images/restore", json={"path": photo, "tool": "faces", "strength": 2})
    assert response.status_code == 422
