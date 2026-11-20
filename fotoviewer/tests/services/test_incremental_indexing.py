import os

import numpy as np
from PIL import Image

from fotoviewer.services.indexing.batch import run_batch_indexing
from fotoviewer.services.indexing.index import Index


def make_image(path, color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path)


class CountingExtract:
    def __init__(self, entries_per_image=1):
        self.calls = []
        self.entries_per_image = entries_per_image

    def __call__(self, image, rel_path):
        self.calls.append(rel_path)
        return [(rel_path, np.array([[float(len(self.calls)), 1.0]]))] * self.entries_per_image


def index(images_dir, out_dir, extract, **kwargs):
    return run_batch_indexing(str(images_dir), str(out_dir), "image", "image.npz", "test", extract, **kwargs)


def load_files(out_dir):
    loaded = Index()
    loaded.load("image", str(out_dir / "image.npz"))
    return sorted(str(f) for f in loaded.files("image"))


def test_second_run_only_processes_new_files(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "a.jpg")
    index(images, out, CountingExtract())

    make_image(images / "b.jpg")
    extract = CountingExtract()
    result = index(images, out, extract)

    assert extract.calls == ["b.jpg"]
    assert result["processed"] == 1 and result["reused"] == 1
    assert load_files(out) == ["a.jpg", "b.jpg"]


def test_unchanged_entries_keep_their_features(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "a.jpg")
    index(images, out, lambda image, rel: [(rel, np.array([[0.25, 0.5]]))])

    make_image(images / "b.jpg")
    index(images, out, lambda image, rel: [(rel, np.array([[9.0, 9.0]]))])

    loaded = Index()
    loaded.load("image", str(out / "image.npz"))
    by_file = dict(zip(map(str, loaded.files("image")), loaded.features("image").tolist()))
    assert by_file["a.jpg"] == [0.25, 0.5]
    assert by_file["b.jpg"] == [9.0, 9.0]


def test_modified_file_is_reprocessed(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "a.jpg")
    make_image(images / "b.jpg")
    index(images, out, CountingExtract())

    make_image(images / "a.jpg", color="blue")
    stat = os.stat(images / "a.jpg")
    os.utime(images / "a.jpg", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    extract = CountingExtract()
    index(images, out, extract)

    assert extract.calls == ["a.jpg"]
    assert load_files(out) == ["a.jpg", "b.jpg"]


def test_deleted_file_is_removed_from_index(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "a.jpg")
    make_image(images / "b.jpg")
    index(images, out, CountingExtract())

    (images / "b.jpg").unlink()
    extract = CountingExtract()
    result = index(images, out, extract)

    assert extract.calls == []
    assert result["removed"] == 1
    assert load_files(out) == ["a.jpg"]


def test_files_without_entries_are_not_reprocessed(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "empty.jpg")
    index(images, out, CountingExtract(entries_per_image=0))

    extract = CountingExtract(entries_per_image=0)
    index(images, out, extract)

    assert extract.calls == []


def test_multiple_entries_per_file_are_all_kept(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "a.jpg")
    index(images, out, CountingExtract(entries_per_image=3))

    make_image(images / "b.jpg")
    index(images, out, CountingExtract(entries_per_image=1))

    assert load_files(out) == ["a.jpg", "a.jpg", "a.jpg", "b.jpg"]


def test_full_rebuild_processes_everything(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "a.jpg")
    index(images, out, CountingExtract())

    extract = CountingExtract()
    index(images, out, extract, full=True)

    assert extract.calls == ["a.jpg"]


def test_model_change_triggers_full_rebuild(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "a.jpg")
    index(images, out, CountingExtract(), model_key="model-a")

    extract = CountingExtract()
    index(images, out, extract, model_key="model-b")

    assert extract.calls == ["a.jpg"]


def test_index_without_manifest_is_rebuilt(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "a.jpg")
    index(images, out, CountingExtract())
    (out / "image.npz.manifest.json").unlink()

    extract = CountingExtract()
    index(images, out, extract)

    assert extract.calls == ["a.jpg"]


def test_unreadable_file_is_retried_on_next_run(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "good.jpg")
    (images / "corrupt.jpg").write_bytes(b"not an image")
    index(images, out, CountingExtract())

    (images / "corrupt.jpg").unlink()
    make_image(images / "corrupt.jpg")
    extract = CountingExtract()
    index(images, out, extract)

    assert extract.calls == ["corrupt.jpg"]
    assert load_files(out) == ["corrupt.jpg", "good.jpg"]


def test_progress_callback_reports_processed_files(tmp_path):
    images, out = tmp_path / "images", tmp_path / "out"
    out.mkdir()
    make_image(images / "a.jpg")
    make_image(images / "b.jpg")

    calls = []
    index(images, out, CountingExtract(), on_progress=lambda done, total: calls.append((done, total)))

    assert calls == [(0, 2), (1, 2), (2, 2)]


def test_creates_missing_output_directory(tmp_path):
    images = tmp_path / "images"
    make_image(images / "a.jpg")

    index(images, tmp_path / "does" / "not" / "exist", CountingExtract())

    assert load_files(tmp_path / "does" / "not" / "exist") == ["a.jpg"]
