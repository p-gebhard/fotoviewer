import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def commands_module(monkeypatch):
    """Import fotoviewer.server.commands with ObjectDetector/ImageProcessor/Index.load
    replaced by mocks, so no real model weights, network access or index file are touched.
    """
    import fotoviewer.object_detector.object_detector as od_mod
    import fotoviewer.image_processor.image_processor as ip_mod
    import fotoviewer.image_processor.index as index_mod

    monkeypatch.setattr(od_mod, "ObjectDetector", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(ip_mod, "ImageProcessor", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(index_mod.Index, "load", lambda self, name, filename: None)

    sys.modules.pop("fotoviewer.server.commands", None)
    module = importlib.import_module("fotoviewer.server.commands")
    yield module
    sys.modules.pop("fotoviewer.server.commands", None)


@pytest.fixture
def app_module(monkeypatch, tmp_path):
    """Import app.py with fotoviewer.server.commands replaced by mocks and
    images_path pointed at an empty tmp_path, so no real models, index or
    personal photo directory are ever touched.
    """
    fake_commands = types.ModuleType("fotoviewer.server.commands")
    fake_commands.detect_objects_in_image = MagicMock(name="detect_objects_in_image")
    fake_commands.search_similar_images = MagicMock(name="search_similar_images")
    fake_commands.search_images_by_text = MagicMock(name="search_images_by_text")
    monkeypatch.setitem(sys.modules, "fotoviewer.server.commands", fake_commands)
    monkeypatch.setenv("FOTOVIEWER_IMAGES_PATH", str(tmp_path))

    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    yield module, fake_commands
    sys.modules.pop("app", None)
