import numpy as np
import pytest
from PIL import Image

from fotoviewer.image_processor.index import Index
from fotoviewer.image_processor.indexer import Indexer


def make_image(path):
    Image.new("RGB", (4, 4), color="red").save(path)


def fake_feature_extractor(image):
    return np.array([[1.0, 0.0]])


def test_index_folder_indexes_supported_images_recursively(tmp_path):
    images_dir = tmp_path / "images"
    sub_dir = images_dir / "sub"
    sub_dir.mkdir(parents=True)

    make_image(images_dir / "a.jpg")
    make_image(sub_dir / "b.png")
    (images_dir / "notes.txt").write_text("not an image")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    indexer = Indexer(fake_feature_extractor)
    indexer.index_folder(str(images_dir), str(output_dir))

    loaded = Index()
    loaded.load("image", str(output_dir / "image.npz"))
    files = sorted(str(f) for f in loaded._loaded_indices["image"]["files"])
    assert files == ["a.jpg", "sub/b.png"]


def test_index_folder_skips_hidden_files(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    make_image(images_dir / "visible.jpg")
    make_image(images_dir / ".hidden.jpg")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    Indexer(fake_feature_extractor).index_folder(str(images_dir), str(output_dir))

    loaded = Index()
    loaded.load("image", str(output_dir / "image.npz"))
    assert list(loaded._loaded_indices["image"]["files"]) == ["visible.jpg"]


def test_index_folder_skips_unreadable_images_without_failing(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    make_image(images_dir / "good.jpg")
    (images_dir / "corrupt.jpg").write_bytes(b"not a real image")

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    Indexer(fake_feature_extractor).index_folder(str(images_dir), str(output_dir))

    loaded = Index()
    loaded.load("image", str(output_dir / "image.npz"))
    assert list(loaded._loaded_indices["image"]["files"]) == ["good.jpg"]


def test_index_folder_raises_for_missing_folder(tmp_path):
    with pytest.raises(ValueError):
        Indexer(fake_feature_extractor).index_folder(str(tmp_path / "does-not-exist"), str(tmp_path))
