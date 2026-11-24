import numpy as np
import pytest

from fotoviewer.services.faces import FACE_SIZE, FFHQ_TEMPLATE, NoFacesFoundError, align_face, restore_faces


def _photo(height=400, width=500):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _landmarks(scale=0.5, offset=(60, 40)):
    return FFHQ_TEMPLATE * scale + np.array(offset, dtype=np.float32)


def test_align_face_maps_landmarks_onto_template():
    face, affine = align_face(_photo(), _landmarks())
    assert face.shape == (FACE_SIZE, FACE_SIZE, 3)
    mapped = np.hstack([_landmarks(), np.ones((5, 1))]) @ affine.T
    assert np.allclose(mapped, FFHQ_TEMPLATE, atol=1.0)


def test_restore_faces_changes_face_area_only():
    photo = _photo()
    result = restore_faces(photo, lambda rgb: [_landmarks()], lambda face: np.full_like(face, 255))

    face_center = _landmarks()[2].astype(int)  # nose
    assert (result[face_center[1], face_center[0]] == 255).all()
    assert np.array_equal(result[-20:, -20:], photo[-20:, -20:])  # far corner untouched
    assert np.array_equal(result.shape, photo.shape)


def test_strength_blends_with_original():
    photo = np.full((400, 500, 3), 100, np.uint8)
    result = restore_faces(photo, lambda rgb: [_landmarks()], lambda face: np.full_like(face, 200), strength=0.5)
    x, y = _landmarks()[2].astype(int)
    assert abs(int(result[y, x, 0]) - 150) <= 2


def test_tiny_faces_are_ignored_and_no_faces_raise():
    with pytest.raises(NoFacesFoundError):
        restore_faces(_photo(), lambda rgb: [_landmarks(scale=0.05)], lambda face: face)
    with pytest.raises(NoFacesFoundError):
        restore_faces(_photo(), lambda rgb: [], lambda face: face)


def test_progress_counts_faces():
    calls = []
    restore_faces(
        _photo(), lambda rgb: [_landmarks(), _landmarks(offset=(200, 100))], lambda f: f, on_progress=lambda *a: calls.append(a)
    )
    assert calls == [(1, 2), (2, 2)]
