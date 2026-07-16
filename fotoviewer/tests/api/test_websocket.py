import base64
import json
from io import BytesIO

from PIL import Image


def _encoded_test_image():
    buf = BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_websocket_dispatches_search_images_by_text(client, fake_services):
    fake_services["image_index"]._results = [{"file": "a.jpg", "score": 0.5}]

    with client.websocket_connect("/echo") as ws:
        ws.send_text(json.dumps({"command": "search_images_by_text", "query": "dog"}))
        response = json.loads(ws.receive_text())

    assert response["status"] == "ok"
    assert response["command"] == "search_images_by_text"
    assert response["response"] == [{"file": "a.jpg", "score": 0.5}]
    fake_services["image_processor"].compute_text_features.assert_called_once_with("dog")


def test_websocket_dispatches_search_similar_images(client, fake_services):
    fake_services["image_index"]._results = [{"file": "b.jpg", "score": 0.4}]

    with client.websocket_connect("/echo") as ws:
        ws.send_text(json.dumps({"command": "search_similar_images", "image": _encoded_test_image()}))
        response = json.loads(ws.receive_text())

    assert response["status"] == "ok"
    assert response["response"] == [{"file": "b.jpg", "score": 0.4}]
    fake_services["image_processor"].compute_image_features.assert_called_once()


def test_websocket_dispatches_detect_objects_in_image(client, fake_services):
    fake_services["object_detector"].detect.return_value = [{"label": "dog", "score": 0.9, "bbox": [1, 2, 3, 4]}]

    with client.websocket_connect("/echo") as ws:
        ws.send_text(json.dumps({"command": "detect_objects_in_image", "image": _encoded_test_image()}))
        response = json.loads(ws.receive_text())

    assert response["status"] == "ok"
    assert response["response"]["detections"] == [
        {"label": "dog", "score": 0.9, "bbox": [1, 2, 3, 4], "identity": None, "identity_score": None}
    ]


def test_websocket_rejects_unknown_command(client):
    with client.websocket_connect("/echo") as ws:
        ws.send_text(json.dumps({"command": "not_a_real_command"}))
        response = json.loads(ws.receive_text())

    assert response["status"] == "error"
    assert "not_a_real_command" in response["reason"]


def test_websocket_rejects_missing_required_field(client):
    with client.websocket_connect("/echo") as ws:
        ws.send_text(json.dumps({"command": "search_images_by_text"}))
        response = json.loads(ws.receive_text())

    assert response["status"] == "error"
    assert "query" in response["reason"]


def test_websocket_reports_command_exceptions_without_crashing(client, fake_services):
    fake_services["image_processor"].compute_image_features.side_effect = ValueError("boom")

    with client.websocket_connect("/echo") as ws:
        ws.send_text(json.dumps({"command": "search_similar_images", "image": _encoded_test_image()}))
        response = json.loads(ws.receive_text())

    assert response["status"] == "error"
    assert "boom" in response["reason"]
