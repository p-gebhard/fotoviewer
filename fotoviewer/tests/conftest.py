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
from fotoviewer.services.identity import INDIVIDUALS_COLLECTION
from fotoviewer.services.indexing.index import Index


class FakeIndex:
    def __init__(self, results=None, num_features=0):
        self._results = results or []
        self.calls = []
        self._num_features = num_features

    def search(self, name, embedding, k):
        self.calls.append((name, embedding, k))
        return self._results

    def exists(self, name):
        return True

    def num_features(self, name):
        return self._num_features


@pytest.fixture
def fake_services():
    """Mocks handed to the app via the fake lifespan below; tests configure
    their return values directly, no real models/network/personal data involved.
    """
    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    return {
        "image_processor": MagicMock(),
        "object_detector": MagicMock(),
        "image_index": FakeIndex(),
        "individuals_index": individuals_index,
    }


@pytest.fixture
def client(tmp_path, fake_services):
    @asynccontextmanager
    async def fake_lifespan(app):
        app.state.image_processor = fake_services["image_processor"]
        app.state.object_detector = fake_services["object_detector"]
        app.state.image_index = fake_services["image_index"]
        app.state.individuals_index = fake_services["individuals_index"]
        yield

    settings = Settings(images_path=str(tmp_path), index_dir=str(tmp_path))
    app = create_app(settings=settings, lifespan=fake_lifespan)

    with TestClient(app) as test_client:
        yield test_client
