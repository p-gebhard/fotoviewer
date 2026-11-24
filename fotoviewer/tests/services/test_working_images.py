import os
import time

import pytest
from PIL import Image

from fotoviewer.services.image_files import ImageNotFoundError
from fotoviewer.services.working_images import WorkingImages


def test_create_and_get(tmp_path):
    store = WorkingImages(str(tmp_path / "edits"))
    meta = store.create("sub/a.jpg", ["denoise"], Image.new("RGB", (20, 10)))

    path, stored = store.get(meta["id"], "sub/a.jpg")

    assert Image.open(path).size == (20, 10)
    assert stored == meta
    assert (meta["width"], meta["height"]) == (20, 10)


@pytest.mark.parametrize("working_id", ["../../etc/passwd", "abc", "0" * 32])
def test_get_rejects_invalid_or_unknown_ids(tmp_path, working_id):
    with pytest.raises(ImageNotFoundError):
        WorkingImages(str(tmp_path)).get(working_id, "a.jpg")


def test_get_rejects_result_of_another_photo(tmp_path):
    store = WorkingImages(str(tmp_path))
    meta = store.create("a.jpg", ["denoise"], Image.new("RGB", (4, 4)))
    with pytest.raises(ImageNotFoundError):
        store.get(meta["id"], "b.jpg")


def test_cleanup_removes_only_old_files(tmp_path):
    store = WorkingImages(str(tmp_path))
    old = store.create("a.jpg", [], Image.new("RGB", (4, 4)))
    new = store.create("a.jpg", [], Image.new("RGB", (4, 4)))
    past = time.time() - 2 * 24 * 3600
    for suffix in (".png", ".json"):
        os.utime(tmp_path / f"{old['id']}{suffix}", (past, past))

    assert store.cleanup() == 2
    with pytest.raises(ImageNotFoundError):
        store.get(old["id"], "a.jpg")
    store.get(new["id"], "a.jpg")
