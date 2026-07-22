import numpy as np
import pytest

from fotoviewer.services.indexing.index import Index


def unit(vec):
    return vec / np.linalg.norm(vec)


def test_exists_reflects_created_and_dropped_indices():
    index = Index()
    assert not index.exists("image")

    index.create("image")
    assert index.exists("image")

    index.drop("image")
    assert not index.exists("image")


def test_add_and_num_features():
    index = Index()
    index.create("image")
    assert index.num_features("image") == 0

    index.add("image", "a.jpg", np.array([[1.0, 0.0]]))
    index.add("image", "b.jpg", np.array([[0.0, 1.0]]))

    assert index.num_features("image") == 2


def test_add_batch_appends_all_entries_at_once():
    index = Index()
    index.create("image")
    index.add_batch("image", ["a.jpg", "b.jpg"], np.array([[1.0, 0.0], [0.0, 1.0]]))

    assert list(index.files("image")) == ["a.jpg", "b.jpg"]
    assert index.features("image").tolist() == [[1.0, 0.0], [0.0, 1.0]]

    index.add_batch("image", ["c.jpg"], np.array([[1.0, 1.0]]))

    assert list(index.files("image")) == ["a.jpg", "b.jpg", "c.jpg"]
    assert index.num_features("image") == 3


def test_files_and_features_expose_stored_entries():
    index = Index()
    index.create("image")
    index.add("image", "a.jpg", np.array([[1.0, 0.0]]))
    index.add("image", "b.jpg", np.array([[0.0, 1.0]]))

    assert list(index.files("image")) == ["a.jpg", "b.jpg"]
    assert index.features("image").tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_search_ranks_by_cosine_similarity():
    index = Index()
    index.create("image")
    index.add("image", "same.jpg", unit(np.array([[1.0, 0.0]])))
    index.add("image", "orthogonal.jpg", unit(np.array([[0.0, 1.0]])))
    index.add("image", "opposite.jpg", unit(np.array([[-1.0, 0.0]])))

    query = unit(np.array([[1.0, 0.0]]))
    results = index.search("image", query, k=3)

    assert [r["file"] for r in results] == ["same.jpg", "orthogonal.jpg", "opposite.jpg"]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.5)
    assert results[2]["score"] == pytest.approx(0.0)


def test_search_respects_k():
    index = Index()
    index.create("image")
    for i in range(5):
        index.add("image", f"{i}.jpg", unit(np.array([[1.0, i * 0.1]])))

    results = index.search("image", unit(np.array([[1.0, 0.0]])), k=2)
    assert len(results) == 2


def test_save_and_load_round_trip(tmp_path):
    index = Index()
    index.create("image")
    index.add("image", "a.jpg", unit(np.array([[1.0, 0.0]])))
    index.add("image", "b.jpg", unit(np.array([[0.0, 1.0]])))

    filename = tmp_path / "image.npz"
    index.save("image", str(filename))

    loaded = Index()
    loaded.load("image", str(filename))

    assert loaded.num_features("image") == 2
    results = loaded.search("image", unit(np.array([[1.0, 0.0]])), k=1)
    assert results[0]["file"] == "a.jpg"
