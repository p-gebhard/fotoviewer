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
from fotoviewer.services.enhance import PhotoEnhancer
from fotoviewer.services.restoration import LoadedModel, Restorer
from fotoviewer.services.working_images import WorkingImages
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


def fake_model_loader(weights_path, device):
    """Stands in for spandrel: an "AI model" that inverts the image (scale 1) or repeats
    pixels (upscaling tools, scale taken from the weights file name)."""
    import torch.nn.functional as F

    scale = 4 if "x4" in weights_path.name else 2 if "x2" in weights_path.name else 1
    if scale == 1:
        # *rest: inpainting models also receive the mask.
        return LoadedModel(fn=lambda x, *rest: 1 - x)
    return LoadedModel(fn=lambda x: F.interpolate(x, scale_factor=scale, mode="nearest"), scale=scale)


def fake_downloader(tool, target, on_progress=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"fake weights")


def fake_face_detector_factory(weights_path):
    """Finds one face (in FFHQ layout, half size) in every image of at least 300×250 px."""
    from fotoviewer.services.faces import FFHQ_TEMPLATE

    def detect(rgb):
        return [FFHQ_TEMPLATE * 0.5 + 20] if rgb.shape[0] >= 250 and rgb.shape[1] >= 300 else []

    return detect


@pytest.fixture
def restorer(tmp_path):
    return Restorer(
        "cpu",
        str(tmp_path / "models"),
        loader=fake_model_loader,
        downloader=fake_downloader,
        face_detector_factory=fake_face_detector_factory,
    )


@pytest.fixture
def language():
    """UI language of the test app; pinned so a local .env cannot change test results.
    Override per test with @pytest.mark.parametrize("language", ["de"])."""
    return "en"


@pytest.fixture
def client(tmp_path, fake_services, language, restorer):
    @asynccontextmanager
    async def fake_lifespan(app):
        app.state.image_processor = fake_services["image_processor"]
        app.state.object_detector = fake_services["object_detector"]
        app.state.image_index = fake_services["image_index"]
        app.state.individuals_index = fake_services["individuals_index"]
        app.state.detections_index = fake_services["detections_index"]
        app.state.date_cache = DateCache()
        app.state.photo_enhancer = PhotoEnhancer("cpu", WorkingImages(str(tmp_path / ".edits")))
        app.state.restorer = restorer
        app.state.import_status = {"running": False, "phase": "idle", "error": None, "copy_result": None}
        app.state.reindex_status = {"running": False, "phase": "idle", "error": None}
        yield

    settings = Settings(images_path=str(tmp_path), index_dir=str(tmp_path), language=language)
    app = create_app(settings=settings, lifespan=fake_lifespan)

    with TestClient(app) as test_client:
        yield test_client
