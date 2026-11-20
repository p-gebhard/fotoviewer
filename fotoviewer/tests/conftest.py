import sys
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fotoviewer.core.config import Settings
from fotoviewer.main import create_app
from fotoviewer.services.date_cache import DateCache
from fotoviewer.services.identity import INDIVIDUALS_COLLECTION
from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION
from fotoviewer.services.indexing.index import Index


@pytest.fixture
def fake_services():
    """Mocks handed to the app via the fake lifespan below; tests configure
    their return values directly, no real models/network/personal data involved.
    image_index/individuals_index/detections_index are the real Index class (pure
    numpy, no external deps) so they always match its actual public API.
    """
    image_index = Index()
    image_index.create("image")

    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)

    detections_index = Index()
    detections_index.create(DETECTIONS_COLLECTION)

    return {
        "image_processor": MagicMock(),
        "object_detector": MagicMock(),
        "image_index": image_index,
        "individuals_index": individuals_index,
        "detections_index": detections_index,
    }


@pytest.fixture
def language():
    """UI language of the test app; pinned so a local .env cannot change test results.
    Override per test with @pytest.mark.parametrize("language", ["de"])."""
    return "en"


@pytest.fixture
def client(tmp_path, fake_services, language):
    @asynccontextmanager
    async def fake_lifespan(app):
        app.state.image_processor = fake_services["image_processor"]
        app.state.object_detector = fake_services["object_detector"]
        app.state.image_index = fake_services["image_index"]
        app.state.individuals_index = fake_services["individuals_index"]
        app.state.detections_index = fake_services["detections_index"]
        app.state.date_cache = DateCache()
        app.state.import_status = {"running": False, "phase": "idle", "error": None, "copy_result": None}
        app.state.reindex_status = {"running": False, "phase": "idle", "error": None}
        yield

    settings = Settings(images_path=str(tmp_path), index_dir=str(tmp_path), language=language)
    app = create_app(settings=settings, lifespan=fake_lifespan)

    with TestClient(app) as test_client:
        yield test_client
