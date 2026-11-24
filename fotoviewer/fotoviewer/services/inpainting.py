"""Object removal: fills a user-painted mask with plausible content (LaMa).

Only a crop around the mask (with surrounding context) is processed. Large crops are
downscaled for the model and the result is scaled back up; in the photo, only the masked
pixels (with a slightly grown, feathered edge) are replaced, so everything else keeps
its original sharpness.
"""

from typing import Callable

import cv2
import numpy as np

# The model sees at most this many pixels on the long side of the crop.
MAX_INPAINT_SIDE = 1024
# Context around the mask's bounding box, relative to its size (and at least MIN_CONTEXT px).
CONTEXT_FACTOR = 0.6
MIN_CONTEXT = 96


class EmptyMaskError(ValueError):
    pass


def _crop_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise EmptyMaskError()
    height, width = mask.shape
    bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    context = max(MIN_CONTEXT, int(CONTEXT_FACTOR * max(bx1 - bx0, by1 - by0)))
    return max(0, bx0 - context), max(0, by0 - context), min(width, bx1 + context), min(height, by1 + context)


def inpaint(
    rgb: np.ndarray,
    mask: np.ndarray,
    model: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """rgb: (H, W, 3) uint8, mask: (H, W) bool (True = remove). model maps an RGB crop
    and its mask (same size) to the filled crop. Raises EmptyMaskError."""
    x0, y0, x1, y1 = _crop_box(mask)
    crop = rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1].astype(np.uint8)

    # Grow the mask a little: painted strokes rarely cover an object's edge and shadow fully.
    grow = max(3, int(0.01 * max(crop_mask.shape)))
    crop_mask = cv2.dilate(crop_mask, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))

    height, width = crop_mask.shape
    factor = min(1.0, MAX_INPAINT_SIDE / max(height, width))
    if factor < 1:
        size = (max(1, round(width * factor)), max(1, round(height * factor)))
        model_crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        model_mask = cv2.resize(crop_mask, size, interpolation=cv2.INTER_NEAREST)
        filled = cv2.resize(model(model_crop, model_mask.astype(bool)), (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        filled = model(crop, crop_mask.astype(bool))

    # Feathered blend so the filled area merges without a hard seam.
    alpha = cv2.GaussianBlur(crop_mask.astype(np.float32), (2 * grow + 1, 2 * grow + 1), 0)
    alpha = np.maximum(alpha, crop_mask)[..., None]  # fully replace inside the mask itself

    result = rgb.copy()
    result[y0:y1, x0:x1] = np.clip(alpha * filled.astype(np.float32) + (1 - alpha) * crop.astype(np.float32), 0, 255).astype(np.uint8)
    return result
