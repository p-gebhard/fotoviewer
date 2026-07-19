import pytest
from PIL import Image

from fotoviewer.services.folder_browser import list_folder
from fotoviewer.services.image_files import ImageNotFoundError


def make_image(path):
    Image.new("RGB", (4, 4), color="red").save(path)


def test_list_folder_returns_subfolders_and_images(tmp_path):
    sub_dir = tmp_path / "Urlaub"
    sub_dir.mkdir()
    make_image(sub_dir / "beach.jpg")
    make_image(tmp_path / "a.jpg")
    make_image(tmp_path / "b.jpg")

    result = list_folder(str(tmp_path), "")

    assert result["path"] == ""
    assert result["breadcrumbs"] == []
    assert result["folders"] == [{"name": "Urlaub", "path": "Urlaub", "image_count": 1}]
    assert sorted(result["images"]) == ["a.jpg", "b.jpg"]


def test_list_folder_builds_breadcrumbs_for_nested_path(tmp_path):
    nested = tmp_path / "Urlaub" / "2019"
    nested.mkdir(parents=True)
    make_image(nested / "photo.jpg")

    result = list_folder(str(tmp_path), "Urlaub/2019")

    assert result["breadcrumbs"] == [
        {"name": "Urlaub", "path": "Urlaub"},
        {"name": "2019", "path": "Urlaub/2019"},
    ]
    assert result["images"] == ["photo.jpg"]


def test_list_folder_skips_hidden_entries(tmp_path):
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    make_image(tmp_path / ".hidden.jpg")
    make_image(tmp_path / "visible.jpg")

    result = list_folder(str(tmp_path), "")

    assert result["folders"] == []
    assert result["images"] == ["visible.jpg"]


def test_list_folder_raises_for_missing_folder(tmp_path):
    with pytest.raises(ImageNotFoundError):
        list_folder(str(tmp_path), "does-not-exist")
