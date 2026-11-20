import json
import os

import pytest
from PIL import ExifTags, Image

from fotoviewer.services.date_cache import DateCache
from fotoviewer.services.image_files import ImageNotFoundError


def make_photo(path, date_time=None, date_time_original=None):
    image = Image.new("RGB", (10, 10))
    exif = image.getexif()
    if date_time:
        exif[0x0132] = date_time
    if date_time_original:
        exif.get_ifd(ExifTags.IFD.Exif)[0x9003] = date_time_original
    image.save(path, exif=exif)


def test_reads_date_time_original_from_exif_sub_ifd(tmp_path):
    make_photo(tmp_path / "a.jpg", date_time="2024:01:01 00:00:00", date_time_original="2019:05:06 07:08:09")

    assert DateCache().get(str(tmp_path), "a.jpg") == "2019:05:06 07:08:09"


def test_falls_back_to_date_time(tmp_path):
    make_photo(tmp_path / "a.jpg", date_time="2020:01:02 03:04:05")

    assert DateCache().get(str(tmp_path), "a.jpg") == "2020:01:02 03:04:05"


def test_returns_none_without_exif_date(tmp_path):
    make_photo(tmp_path / "a.jpg")

    assert DateCache().get(str(tmp_path), "a.jpg") is None


def test_returns_none_for_unreadable_file(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"garbage")

    assert DateCache().get(str(tmp_path), "a.jpg") is None


def test_raises_for_missing_file(tmp_path):
    with pytest.raises(ImageNotFoundError):
        DateCache().get(str(tmp_path), "missing.jpg")


def test_persists_and_reuses_cached_dates(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    make_photo(images / "a.jpg", date_time="2020:01:02 03:04:05")
    cache_file = str(tmp_path / "dates.json")

    cache = DateCache(cache_file)
    cache.get(str(images), "a.jpg")
    cache.save()

    # Tamper with the cached value: a reload must serve it without re-reading the photo.
    data = json.loads((tmp_path / "dates.json").read_text())
    data["a.jpg"][2] = "1999:01:01 00:00:00"
    (tmp_path / "dates.json").write_text(json.dumps(data))

    assert DateCache(cache_file).get(str(images), "a.jpg") == "1999:01:01 00:00:00"


def test_modified_photo_is_reread(tmp_path):
    make_photo(tmp_path / "a.jpg", date_time="2020:01:02 03:04:05")
    cache = DateCache()
    cache.get(str(tmp_path), "a.jpg")

    make_photo(tmp_path / "a.jpg", date_time="2021:01:02 03:04:05")
    stat = os.stat(tmp_path / "a.jpg")
    os.utime(tmp_path / "a.jpg", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert cache.get(str(tmp_path), "a.jpg") == "2021:01:02 03:04:05"


def test_prune_drops_entries_of_removed_photos(tmp_path):
    make_photo(tmp_path / "a.jpg", date_time="2020:01:02 03:04:05")
    make_photo(tmp_path / "b.jpg", date_time="2020:01:02 03:04:05")
    cache_file = str(tmp_path / "dates.json")
    cache = DateCache(cache_file)
    cache.warm(str(tmp_path), ["a.jpg", "b.jpg"])

    cache.prune({"a.jpg"})
    cache.save()

    assert set(json.loads((tmp_path / "dates.json").read_text())) == {"a.jpg"}


def test_corrupt_cache_file_is_ignored(tmp_path):
    (tmp_path / "dates.json").write_text("{not json")
    make_photo(tmp_path / "a.jpg", date_time="2020:01:02 03:04:05")

    assert DateCache(str(tmp_path / "dates.json")).get(str(tmp_path), "a.jpg") == "2020:01:02 03:04:05"
