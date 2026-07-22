import numpy as np
import pytest

from fotoviewer.services import individuals_store
from fotoviewer.services.identity import INDIVIDUALS_COLLECTION
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.individuals_store import IndividualEntryNotFoundError


def make_index_with_two_entries():
    index = Index()
    index.create(INDIVIDUALS_COLLECTION)
    index.add_batch(INDIVIDUALS_COLLECTION, ["Rex", "Rex"], np.array([[1.0, 0.0], [0.9, 0.1]]))
    return index


def test_append_entry_backfills_placeholders_for_pre_existing_entries(tmp_path):
    index = make_index_with_two_entries()
    index.add(INDIVIDUALS_COLLECTION, "Rex", np.array([[0.8, 0.2]]))

    entry = individuals_store.append_entry(str(tmp_path), index, "photos/rex3.jpg", [1, 2, 3, 4])

    assert entry["path"] == "photos/rex3.jpg"
    assert entry["bbox"] == [1, 2, 3, 4]
    groups = individuals_store.list_individuals(str(tmp_path), index)
    assert len(groups) == 1
    assert len(groups[0]["entries"]) == 3
    assert groups[0]["entries"][0]["path"] is None
    assert groups[0]["entries"][2]["path"] == "photos/rex3.jpg"


def test_list_individuals_groups_by_name(tmp_path):
    index = Index()
    index.create(INDIVIDUALS_COLLECTION)
    index.add_batch(INDIVIDUALS_COLLECTION, ["Rex", "Mia"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    individuals_store._save_metadata(str(tmp_path), [
        {"id": "a", "path": "rex.jpg", "bbox": [0, 0, 1, 1]},
        {"id": "b", "path": "mia.jpg", "bbox": [0, 0, 1, 1]},
    ])

    groups = individuals_store.list_individuals(str(tmp_path), index)

    assert {g["name"] for g in groups} == {"Rex", "Mia"}


def test_list_individuals_returns_empty_for_missing_collection(tmp_path):
    index = Index()

    assert individuals_store.list_individuals(str(tmp_path), index) == []


def test_rename_entry_relabels_only_that_entry(tmp_path):
    index = make_index_with_two_entries()
    individuals_store._save_metadata(str(tmp_path), [
        {"id": "a", "path": "1.jpg", "bbox": [0, 0, 1, 1]},
        {"id": "b", "path": "2.jpg", "bbox": [0, 0, 1, 1]},
    ])

    individuals_store.rename_entry(str(tmp_path), index, "b", "Rex (falsch)")

    names = list(index.files(INDIVIDUALS_COLLECTION))
    assert names == ["Rex", "Rex (falsch)"]
    assert index.num_features(INDIVIDUALS_COLLECTION) == 2


def test_rename_entry_raises_for_unknown_id(tmp_path):
    index = make_index_with_two_entries()

    with pytest.raises(IndividualEntryNotFoundError):
        individuals_store.rename_entry(str(tmp_path), index, "does-not-exist", "New")


def test_delete_entry_removes_matching_feature_and_metadata(tmp_path):
    index = make_index_with_two_entries()
    individuals_store._save_metadata(str(tmp_path), [
        {"id": "a", "path": "1.jpg", "bbox": [0, 0, 1, 1]},
        {"id": "b", "path": "2.jpg", "bbox": [0, 0, 1, 1]},
    ])

    individuals_store.delete_entry(str(tmp_path), index, "a")

    assert index.num_features(INDIVIDUALS_COLLECTION) == 1
    assert list(index.files(INDIVIDUALS_COLLECTION)) == ["Rex"]
    remaining = individuals_store.list_individuals(str(tmp_path), index)
    assert remaining[0]["entries"][0]["id"] == "b"


def test_delete_entry_raises_for_unknown_id(tmp_path):
    index = make_index_with_two_entries()

    with pytest.raises(IndividualEntryNotFoundError):
        individuals_store.delete_entry(str(tmp_path), index, "does-not-exist")


def test_rename_group_relabels_all_entries_with_that_name(tmp_path):
    index = make_index_with_two_entries()

    individuals_store.rename_group(str(tmp_path), index, "Rex", "Rex II")

    assert list(index.files(INDIVIDUALS_COLLECTION)) == ["Rex II", "Rex II"]


def test_delete_group_removes_all_entries_with_that_name(tmp_path):
    index = Index()
    index.create(INDIVIDUALS_COLLECTION)
    index.add_batch(INDIVIDUALS_COLLECTION, ["Rex", "Mia", "Rex"], np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]))

    individuals_store.delete_group(str(tmp_path), index, "Rex")

    assert list(index.files(INDIVIDUALS_COLLECTION)) == ["Mia"]
    assert index.num_features(INDIVIDUALS_COLLECTION) == 1


def test_delete_group_leaves_empty_collection_when_all_removed(tmp_path):
    index = make_index_with_two_entries()

    individuals_store.delete_group(str(tmp_path), index, "Rex")

    assert index.num_features(INDIVIDUALS_COLLECTION) == 0


def test_get_entry_returns_metadata(tmp_path):
    index = make_index_with_two_entries()
    individuals_store._save_metadata(str(tmp_path), [
        {"id": "a", "path": "1.jpg", "bbox": [0, 0, 1, 1]},
        {"id": "b", "path": "2.jpg", "bbox": [0, 0, 1, 1]},
    ])

    entry = individuals_store.get_entry(str(tmp_path), index, "b")

    assert entry["path"] == "2.jpg"


def test_get_entry_raises_for_unknown_id(tmp_path):
    index = make_index_with_two_entries()

    with pytest.raises(IndividualEntryNotFoundError):
        individuals_store.get_entry(str(tmp_path), index, "does-not-exist")
