import pytest
from PIL import Image

from fotoviewer.services.image_files import ImageNotFoundError
from fotoviewer.services.image_metadata import get_image_metadata


def test_get_image_metadata_returns_dimensions_and_format(tmp_path):
    Image.new("RGB", (30, 40)).save(tmp_path / "a.png")

    metadata = get_image_metadata(str(tmp_path), "a.png")

    assert metadata["width"] == 30
    assert metadata["height"] == 40
    assert metadata["format"] == "PNG"
    assert metadata["file_size_bytes"] > 0


def test_get_image_metadata_extracts_date_taken(tmp_path):
    image = Image.new("RGB", (10, 10))
    exif = image.getexif()
    exif[306] = "2003:07:12 10:00:00"  # DateTime
    path = tmp_path / "b.jpg"
    image.save(path, exif=exif)

    metadata = get_image_metadata(str(tmp_path), "b.jpg")

    assert metadata["date_taken"] == "2003:07:12 10:00:00"


def test_get_image_metadata_raises_for_missing_file(tmp_path):
    with pytest.raises(ImageNotFoundError):
        get_image_metadata(str(tmp_path), "missing.jpg")


def test_get_image_metadata_blocks_path_traversal(tmp_path):
    outside_dir = tmp_path.parent / "outside_secret"
    outside_dir.mkdir(exist_ok=True)
    Image.new("RGB", (5, 5)).save(outside_dir / "secret.jpg")

    with pytest.raises(ImageNotFoundError):
        get_image_metadata(str(tmp_path), "../outside_secret/secret.jpg")
