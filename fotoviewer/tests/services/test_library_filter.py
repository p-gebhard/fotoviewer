import numpy as np
from PIL import Image

from fotoviewer.services.identity import INDIVIDUALS_COLLECTION
from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.library_filter import filter_library, list_folder_options


def unit(vec):
    return vec / np.linalg.norm(vec)


def make_image_index(files):
    index = Index()
    index.create("image")
    index.add_batch("image", files, np.array([[1.0, 0.0]] * len(files)))
    return index


def make_dated_photo(images_dir, name, date_taken):
    path = images_dir / name
    image = Image.new("RGB", (10, 10))
    exif = image.getexif()
    exif[306] = date_taken  # DateTime
    image.save(path, exif=exif)


def test_filter_library_returns_all_files_without_filters():
    image_index = make_image_index(["a.jpg", "Urlaub/b.jpg"])

    result = filter_library(image_index, Index(), Index(), "/unused", 0.75)

    assert result == ["Urlaub/b.jpg", "a.jpg"]


def test_filter_library_filters_by_folder():
    image_index = make_image_index(["a.jpg", "Urlaub/b.jpg", "Urlaub/Tag1/c.jpg", "Andere/d.jpg"])

    result = filter_library(image_index, Index(), Index(), "/unused", 0.75, folder="Urlaub")

    assert result == ["Urlaub/Tag1/c.jpg", "Urlaub/b.jpg"]


def test_list_folder_options_returns_unique_sorted_folders():
    image_index = make_image_index(["a.jpg", "Urlaub/b.jpg", "Urlaub/Tag1/c.jpg", "Andere/d.jpg"])

    assert list_folder_options(image_index) == ["Andere", "Urlaub", "Urlaub/Tag1"]


def test_filter_library_filters_by_individual_using_averaged_matching():
    image_index = make_image_index(["a.jpg", "b.jpg"])

    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))

    detections_index = Index()
    detections_index.create(DETECTIONS_COLLECTION)
    detections_index.add_batch(
        DETECTIONS_COLLECTION,
        ["a.jpg", "b.jpg"],
        np.array([unit(np.array([1.0, 0.0])), unit(np.array([0.0, 1.0]))]),  # a matches Rex, b doesn't
    )

    result = filter_library(image_index, individuals_index, detections_index, "/unused", 0.75, individual="Rex")

    assert result == ["a.jpg"]


def test_filter_library_individual_immediately_reflects_new_reference_without_reindexing():
    image_index = make_image_index(["a.jpg"])
    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)

    detections_index = Index()
    detections_index.create(DETECTIONS_COLLECTION)
    detections_index.add(DETECTIONS_COLLECTION, "a.jpg", unit(np.array([[1.0, 0.0]])))

    # Before any reference is tagged, nothing matches.
    assert filter_library(image_index, individuals_index, detections_index, "/unused", 0.75, individual="Rex") == []

    # Tagging a new reference (no detection re-indexing) makes it immediately findable.
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))

    assert filter_library(image_index, individuals_index, detections_index, "/unused", 0.75, individual="Rex") == ["a.jpg"]


def test_filter_library_returns_empty_when_detections_index_has_no_entries():
    image_index = make_image_index(["a.jpg"])
    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))

    result = filter_library(image_index, individuals_index, Index(), "/unused", 0.75, individual="Rex")

    assert result == []


def test_filter_library_filters_by_year_range(tmp_path):
    make_dated_photo(tmp_path, "old.jpg", "1999:01:01 10:00:00")
    make_dated_photo(tmp_path, "target.jpg", "2003:07:12 10:00:00")
    make_dated_photo(tmp_path, "new.jpg", "2010:01:01 10:00:00")
    image_index = make_image_index(["old.jpg", "target.jpg", "new.jpg"])

    result = filter_library(image_index, Index(), Index(), str(tmp_path), 0.75, year_from=2000, year_to=2005)

    assert result == ["target.jpg"]


def test_filter_library_excludes_photos_without_date_when_year_filter_active(tmp_path):
    Image.new("RGB", (10, 10)).save(tmp_path / "no_exif.jpg")
    image_index = make_image_index(["no_exif.jpg"])

    result = filter_library(image_index, Index(), Index(), str(tmp_path), 0.75, year_from=2000)

    assert result == []


def test_filter_library_combines_folder_and_individual_filters():
    image_index = make_image_index(["Urlaub/a.jpg", "Andere/a.jpg"])
    individuals_index = Index()
    individuals_index.create(INDIVIDUALS_COLLECTION)
    individuals_index.add(INDIVIDUALS_COLLECTION, "Rex", unit(np.array([[1.0, 0.0]])))

    detections_index = Index()
    detections_index.create(DETECTIONS_COLLECTION)
    detections_index.add_batch(
        DETECTIONS_COLLECTION,
        ["Urlaub/a.jpg", "Andere/a.jpg"],
        np.array([unit(np.array([1.0, 0.0])), unit(np.array([1.0, 0.0]))]),
    )

    result = filter_library(
        image_index, individuals_index, detections_index, "/unused", 0.75, folder="Urlaub", individual="Rex"
    )

    assert result == ["Urlaub/a.jpg"]
