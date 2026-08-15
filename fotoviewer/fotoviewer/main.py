from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fotoviewer.api.routes import collection_import, detection, individuals, metadata, pages, websocket
from fotoviewer.core.config import Settings
from fotoviewer.services.identity import INDIVIDUALS_COLLECTION
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.object_detector.object_detector import ObjectDetector

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    device = settings.device or ("cuda" if torch.cuda.is_available() else "cpu")

    app.state.image_processor = ImageProcessor(settings.embedding_model_name)
    app.state.object_detector = ObjectDetector(device, settings.detection_model_name, settings.detection_threshold)
    app.state.image_index = Index()
    app.state.image_index.load("image", str(Path(settings.index_dir) / "image.npz"))

    app.state.individuals_index = Index()
    app.state.individuals_index.load_or_create(INDIVIDUALS_COLLECTION, str(Path(settings.index_dir) / "individuals.npz"))

    app.state.detections_index = Index()
    app.state.detections_index.load_or_create(DETECTIONS_COLLECTION, str(Path(settings.index_dir) / "detections.npz"))

    app.state.import_status = {"running": False, "phase": "idle", "error": None, "copy_result": None}
    yield


def create_app(settings: Settings | None = None, lifespan=lifespan) -> FastAPI:
    settings = settings or Settings()

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.state.templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.mount("/images", StaticFiles(directory=settings.images_path), name="images")

    app.include_router(pages.router)
    app.include_router(metadata.router)
    app.include_router(detection.router)
    app.include_router(individuals.router)
    app.include_router(collection_import.router)
    app.include_router(websocket.router)
    return app


app = create_app()
