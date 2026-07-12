import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from fotoviewer.server.commands import detect_objects_in_image, search_similar_images, search_images_by_text


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
images_path = os.environ.get("FOTOVIEWER_IMAGES_PATH", "/home/user/Pictures/personal/")

app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static_content", StaticFiles(directory=str(BASE_DIR / "static_content")), name="static_content")
app.mount("/images", StaticFiles(directory=images_path), name="images")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.websocket("/echo")
async def echo(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            response = {
                "status": "ok",
                "command": None,
                "response": None
            }
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                command = data.get("command")
                if command == "detect_objects_in_image":
                    response["response"] = await run_in_threadpool(detect_objects_in_image, data)
                    response["command"] = command
                elif command == "search_similar_images":
                    response["response"] = await run_in_threadpool(search_similar_images, data)
                    response["command"] = command
                elif command == "search_images_by_text":
                    response["response"] = await run_in_threadpool(search_images_by_text, data)
                    response["command"] = command
                else:
                    response = {
                        "status": "error",
                        "reason": f"Unknown command {command}"
                    }
            except Exception as e:
                logger.error(e)
                response = {
                    "status": "error",
                    "reason": f"Exception: {e}"
                }
            await websocket.send_text(json.dumps(response))
    except WebSocketDisconnect:
        pass
