import pytest
from pydantic import ValidationError

from fotoviewer.core.config import Settings


def test_settings_requires_images_path(monkeypatch):
    monkeypatch.delenv("FOTOVIEWER_IMAGES_PATH", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_relative_paths_are_resolved_against_project_root(monkeypatch, tmp_path):
    from fotoviewer.core.config import PROJECT_ROOT

    monkeypatch.chdir(tmp_path)  # must not depend on the working directory
    settings = Settings(_env_file=None, images_path="photos", index_dir="../indices")

    assert settings.images_path == str(PROJECT_ROOT / "photos")
    assert settings.index_dir == str((PROJECT_ROOT / ".." / "indices").resolve())


def test_absolute_paths_are_kept(tmp_path):
    settings = Settings(_env_file=None, images_path=str(tmp_path), index_dir=str(tmp_path / "idx"))

    assert settings.images_path == str(tmp_path)
    assert settings.index_dir == str(tmp_path / "idx")
