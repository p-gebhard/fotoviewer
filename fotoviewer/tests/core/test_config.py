import pytest
from pydantic import ValidationError

from fotoviewer.core.config import Settings


def test_settings_requires_images_path(monkeypatch):
    monkeypatch.delenv("FOTOVIEWER_IMAGES_PATH", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
