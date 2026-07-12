import json

from starlette.testclient import TestClient


def test_index_route_renders_template(app_module):
    module, _ = app_module
    client = TestClient(module.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "Search by Text" in response.text


def test_static_content_is_served(app_module):
    module, _ = app_module
    client = TestClient(module.app)

    response = client.get("/static_content/main.css")

    assert response.status_code == 200


def test_websocket_dispatches_search_images_by_text(app_module):
    module, fake_commands = app_module
    fake_commands.search_images_by_text.return_value = [{"file": "a.jpg", "score": 0.5}]
    client = TestClient(module.app)

    with client.websocket_connect("/echo") as ws:
        ws.send_text(json.dumps({"command": "search_images_by_text", "query": "dog"}))
        response = json.loads(ws.receive_text())

    assert response["status"] == "ok"
    assert response["command"] == "search_images_by_text"
    assert response["response"] == [{"file": "a.jpg", "score": 0.5}]
    fake_commands.search_images_by_text.assert_called_once_with(
        {"command": "search_images_by_text", "query": "dog"}
    )


def test_websocket_dispatches_search_similar_images(app_module):
    module, fake_commands = app_module
    fake_commands.search_similar_images.return_value = [{"file": "b.jpg", "score": 0.4}]
    client = TestClient(module.app)

    with client.websocket_connect("/echo") as ws:
        ws.send_text(json.dumps({"command": "search_similar_images", "image": "base64=="}))
        response = json.loads(ws.receive_text())

    assert response["status"] == "ok"
    assert response["response"] == [{"file": "b.jpg", "score": 0.4}]


def test_websocket_rejects_unknown_command(app_module):
    module, _ = app_module
    client = TestClient(module.app)

    with client.websocket_connect("/echo") as ws:
        ws.send_text(json.dumps({"command": "not_a_real_command"}))
        response = json.loads(ws.receive_text())

    assert response["status"] == "error"
    assert "not_a_real_command" in response["reason"]


def test_websocket_reports_command_exceptions_without_crashing(app_module):
    module, fake_commands = app_module
    fake_commands.search_similar_images.side_effect = ValueError("boom")
    client = TestClient(module.app)

    with client.websocket_connect("/echo") as ws:
        ws.send_text(json.dumps({"command": "search_similar_images", "image": "x"}))
        response = json.loads(ws.receive_text())

    assert response["status"] == "error"
    assert "boom" in response["reason"]
