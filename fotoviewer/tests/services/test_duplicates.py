import math

import numpy as np

from fotoviewer.services.duplicates import find_duplicate_groups
from fotoviewer.services.indexing.index import Index


def unit(vec):
    return vec / np.linalg.norm(vec)


def build_index(entries):
    index = Index()
    index.create("image")
    for name, vector in entries:
        index.add("image", name, unit(np.array([vector])))
    return index


def test_find_duplicate_groups_groups_near_identical_images():
    index = build_index([
        ("a.jpg", [1.0, 0.0]),
        ("b.jpg", [0.999, 0.001]),
        ("c.jpg", [0.0, 1.0]),
    ])

    groups = find_duplicate_groups(index, threshold=0.95)

    assert len(groups) == 1
    assert sorted(entry["file"] for entry in groups[0]) == ["a.jpg", "b.jpg"]


def test_find_duplicate_groups_chains_transitively():
    # a-b and b-c are ~20 degrees apart each (well above the threshold), but
    # a-c is ~40 degrees apart (below threshold) -> still one group via b.
    def at_angle(degrees):
        radians = math.radians(degrees)
        return [math.cos(radians), math.sin(radians)]

    index = build_index([
        ("a.jpg", at_angle(0)),
        ("b.jpg", at_angle(20)),
        ("c.jpg", at_angle(40)),
    ])

    groups = find_duplicate_groups(index, threshold=0.95)

    assert len(groups) == 1
    assert sorted(entry["file"] for entry in groups[0]) == ["a.jpg", "b.jpg", "c.jpg"]


def test_find_duplicate_groups_excludes_singletons():
    index = build_index([
        ("a.jpg", [1.0, 0.0]),
        ("b.jpg", [0.0, 1.0]),
    ])

    groups = find_duplicate_groups(index, threshold=0.95)

    assert groups == []


def test_find_duplicate_groups_returns_empty_for_empty_index():
    assert find_duplicate_groups(Index(), threshold=0.95) == []
