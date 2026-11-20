import os

import pytest
from PIL import Image

from fotoviewer.services.image_files import ImageNotFoundError
from fotoviewer.services.thumbnails import get_thumbnail


def test_creates_downscaled_jpeg_keeping_aspect_ratio(tmp_path):
    Image.new("RGBA", (1000, 500)).save(tmp_path / "a.png")

    thumb = get_thumbnail(str(tmp_path), str(tmp_path / "cache"), "a.png", size=200)

    with Image.open(thumb) as image:
        assert image.format == "JPEG"
        assert image.size == (200, 100)


def test_reuses_cached_thumbnail(tmp_path):
    Image.new("RGB", (100, 100)).save(tmp_path / "a.jpg")
    first = get_thumbnail(str(tmp_path), str(tmp_path / "cache"), "a.jpg")
    mtime = first.stat().st_mtime_ns

    second = get_thumbnail(str(tmp_path), str(tmp_path / "cache"), "a.jpg")

    assert second == first
    assert second.stat().st_mtime_ns == mtime


def test_modified_photo_gets_new_thumbnail(tmp_path):
    Image.new("RGB", (100, 100)).save(tmp_path / "a.jpg")
    first = get_thumbnail(str(tmp_path), str(tmp_path / "cache"), "a.jpg")

    Image.new("RGB", (100, 50)).save(tmp_path / "a.jpg")
    stat = os.stat(tmp_path / "a.jpg")
    os.utime(tmp_path / "a.jpg", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert get_thumbnail(str(tmp_path), str(tmp_path / "cache"), "a.jpg") != first


def test_applies_exif_orientation(tmp_path):
    image = Image.new("RGB", (200, 100))
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90° clockwise when displayed
    image.save(tmp_path / "a.jpg", exif=exif)

    thumb = get_thumbnail(str(tmp_path), str(tmp_path / "cache"), "a.jpg")

    with Image.open(thumb) as result:
        assert result.size == (100, 200)


def test_rejects_paths_outside_library(tmp_path):
    with pytest.raises(ImageNotFoundError):
        get_thumbnail(str(tmp_path), str(tmp_path / "cache"), "../etc/passwd")
