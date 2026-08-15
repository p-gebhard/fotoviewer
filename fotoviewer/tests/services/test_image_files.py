import time

import pytest
from PIL import Image

from fotoviewer.services import image_files
from fotoviewer.services.image_files import (
    ImageNotFoundError,
    find_image_paths,
    resolve_directory_path,
    resolve_or_create_directory_path,
)


def make_image(path):
    Image.new("RGB", (4, 4), color="red").save(path)


def test_find_image_paths_finds_images_recursively_and_skips_hidden(tmp_path):
    sub_dir = tmp_path / "sub"
    hidden_dir = tmp_path / ".hidden"
    sub_dir.mkdir()
    hidden_dir.mkdir()
    make_image(tmp_path / "a.jpg")
    make_image(sub_dir / "b.png")
    make_image(hidden_dir / "c.jpg")
    make_image(tmp_path / ".d.jpg")
    (tmp_path / "notes.txt").write_text("not an image")

    paths = find_image_paths(str(tmp_path))

    names = sorted(p.split("/")[-1] for p in paths)
    assert names == ["a.jpg", "b.png"]


def test_resolve_directory_path_returns_existing_directory(tmp_path):
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()

    resolved = resolve_directory_path(str(tmp_path), "sub")

    assert resolved == sub_dir.resolve()


def test_resolve_directory_path_raises_for_file(tmp_path):
    make_image(tmp_path / "a.jpg")

    with pytest.raises(ImageNotFoundError):
        resolve_directory_path(str(tmp_path), "a.jpg")


def test_resolve_directory_path_blocks_path_traversal(tmp_path):
    outside_dir = tmp_path.parent / "outside_secret_dir"
    outside_dir.mkdir(exist_ok=True)

    with pytest.raises(ImageNotFoundError):
        resolve_directory_path(str(tmp_path), "../outside_secret_dir")


def test_find_image_paths_caches_result_within_ttl(tmp_path, monkeypatch):
    image_files._cache.clear()
    make_image(tmp_path / "a.jpg")

    first = find_image_paths(str(tmp_path))
    make_image(tmp_path / "b.jpg")
    second = find_image_paths(str(tmp_path))

    assert first == second

    future = time.monotonic() + image_files._CACHE_TTL_SECONDS + 1
    monkeypatch.setattr(image_files.time, "monotonic", lambda: future)
    third = find_image_paths(str(tmp_path))

    assert len(third) == 2


def test_resolve_or_create_directory_path_creates_missing_directory(tmp_path):
    resolved = resolve_or_create_directory_path(str(tmp_path), "new/nested")

    assert resolved.is_dir()
    assert resolved == (tmp_path / "new" / "nested").resolve()


def test_resolve_or_create_directory_path_returns_existing_directory(tmp_path):
    sub_dir = tmp_path / "sub"
    sub_dir.mkdir()

    resolved = resolve_or_create_directory_path(str(tmp_path), "sub")

    assert resolved == sub_dir.resolve()


def test_resolve_or_create_directory_path_blocks_path_traversal(tmp_path):
    with pytest.raises(ImageNotFoundError):
        resolve_or_create_directory_path(str(tmp_path), "../outside_new_dir")


def test_invalidate_clears_cached_entry(tmp_path):
    image_files._cache.clear()
    make_image(tmp_path / "a.jpg")
    find_image_paths(str(tmp_path))
    assert str(tmp_path) in image_files._cache

    image_files.invalidate(str(tmp_path))

    assert str(tmp_path) not in image_files._cache
