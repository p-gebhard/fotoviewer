import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fotoviewer.api.routes import (
    collection_import,
    detection,
    duplicates,
    enhance,
    individuals,
    metadata,
    pages,
    reindex,
    thumbnails,
    websocket,
)
from fotoviewer.core.config import Settings
from fotoviewer.core.i18n import Translator
from fotoviewer.services.date_cache import DATE_CACHE_FILENAME, DateCache
from fotoviewer.services.enhance import PhotoEnhancer
from fotoviewer.services.restoration import Restorer
from fotoviewer.services.working_images import WORKING_DIRNAME, WorkingImages
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
    # Start with empty indices on a fresh install; they are built via the CLI or /sources.
    Path(settings.index_dir).mkdir(parents=True, exist_ok=True)
    app.state.image_index.load_or_create("image", str(Path(settings.index_dir) / "image.npz"))

    app.state.individuals_index = Index()
    app.state.individuals_index.load_or_create(INDIVIDUALS_COLLECTION, str(Path(settings.index_dir) / "individuals.npz"))

    app.state.detections_index = Index()
    app.state.detections_index.load_or_create(DETECTIONS_COLLECTION, str(Path(settings.index_dir) / "detections.npz"))

    working_images = WorkingImages(str(Path(settings.index_dir) / WORKING_DIRNAME))
    working_images.cleanup()
    app.state.photo_enhancer = PhotoEnhancer(device, working_images)
    app.state.restorer = Restorer(device, settings.model_dir)
    app.state.date_cache = DateCache(str(Path(settings.index_dir) / DATE_CACHE_FILENAME))

    app.state.import_status = {"running": False, "phase": "idle", "error": None, "copy_result": None}
    app.state.reindex_status = {"running": False, "phase": "idle", "error": None}
    yield


def create_app(settings: Settings | None = None, lifespan=lifespan) -> FastAPI:
    settings = settings or Settings()

    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    app.state.translator = Translator(settings.language)
    # Only one AI restoration job at a time — they need a lot of GPU memory.
    app.state.restore_status = {"running": False, "phase": "idle", "error": None, "result": None}
    app.state.restore_lock = threading.Lock()
    app.state.templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.state.templates.env.globals.update(
        t=app.state.translator,
        language=settings.language,
        js_messages=app.state.translator.js_messages(),
    )

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.mount("/images", StaticFiles(directory=settings.images_path), name="images")

    app.include_router(pages.router)
    app.include_router(metadata.router)
    app.include_router(detection.router)
    app.include_router(individuals.router)
    app.include_router(duplicates.router)
    app.include_router(collection_import.router)
    app.include_router(reindex.router)
    app.include_router(thumbnails.router)
    app.include_router(enhance.router)
    app.include_router(websocket.router)
    return app


app = create_app()
