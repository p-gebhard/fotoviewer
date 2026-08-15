import numpy as np


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise similarity between L2-normalized rows of a and b, rescaled from [-1, 1] to [0, 1]."""
    return (1 + np.dot(a, b.T)) / 2
