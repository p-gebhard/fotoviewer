import numpy as np
import pytest

from fotoviewer.services.similarity import cosine_scores


def unit(vec):
    return vec / np.linalg.norm(vec)


def test_cosine_scores_identical_vectors_score_one():
    a = unit(np.array([[1.0, 0.0]]))
    scores = cosine_scores(a, a)

    assert scores[0, 0] == pytest.approx(1.0)


def test_cosine_scores_orthogonal_vectors_score_half():
    a = unit(np.array([[1.0, 0.0]]))
    b = unit(np.array([[0.0, 1.0]]))

    assert cosine_scores(a, b)[0, 0] == pytest.approx(0.5)


def test_cosine_scores_opposite_vectors_score_zero():
    a = unit(np.array([[1.0, 0.0]]))
    b = unit(np.array([[-1.0, 0.0]]))

    assert cosine_scores(a, b)[0, 0] == pytest.approx(0.0)


def test_cosine_scores_returns_pairwise_matrix_for_many_vs_many():
    # Rows are already unit-length; unit() normalizes the whole matrix (Frobenius
    # norm) rather than per-row, so it must not be used here.
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    scores = cosine_scores(a, b)

    assert scores.shape == (2, 3)
    assert scores[0, 0] == pytest.approx(1.0)
    assert scores[1, 2] == pytest.approx(0.5)
