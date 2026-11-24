"""Face restoration: detect faces, align each to the 512×512 FFHQ layout, restore it with
a face model (GFPGAN) and paste it back with a feathered mask.

Follows the approach of facexlib's FaceRestoreHelper (used by GFPGAN itself), with face
detection by OpenCV's YuNet, which also provides the five landmarks for alignment.
"""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

FACE_SIZE = 512
# Positions of eyes, nose tip and mouth corners (image left to right) in an aligned
# 512×512 FFHQ face — the layout GFPGAN was trained on.
FFHQ_TEMPLATE = np.array(
    [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]],
    dtype=np.float32,
)
# Long side of the copy the detector runs on; faces are found reliably at this size and
# detection stays fast on the CPU.
DETECTION_MAX_SIDE = 1600
DETECTION_SCORE = 0.75
# Faces with an eye distance below this (in pixels of the full photo) are too small to
# restore meaningfully.
MIN_EYE_DISTANCE = 12

# (H, W, 3) uint8 RGB -> list of (5, 2) landmark arrays in pixel coordinates.
FaceDetector = Callable[[np.ndarray], list[np.ndarray]]


class NoFacesFoundError(Exception):
    pass


def yunet_detector(weights_path: Path) -> FaceDetector:
    def detect(rgb: np.ndarray) -> list[np.ndarray]:
        height, width = rgb.shape[:2]
        factor = min(1.0, DETECTION_MAX_SIDE / max(height, width))
        small = cv2.resize(rgb, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA) if factor < 1 else rgb
        detector = cv2.FaceDetectorYN.create(str(weights_path), "", (small.shape[1], small.shape[0]), DETECTION_SCORE)
        _, faces = detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if faces is None:
            return []
        # Columns: box (4), landmarks right eye, left eye, nose, right/left mouth corner (10), score.
        return [face[4:14].reshape(5, 2) / factor for face in faces]

    return detect


def align_face(rgb: np.ndarray, landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (aligned 512×512 face, affine matrix photo -> face)."""
    affine, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), FFHQ_TEMPLATE, method=cv2.LMEDS)
    face = cv2.warpAffine(rgb, affine, (FACE_SIZE, FACE_SIZE), borderMode=cv2.BORDER_REFLECT_101)
    return face, affine


def paste_face(rgb: np.ndarray, restored: np.ndarray, affine: np.ndarray) -> None:
    """Warps the restored face back into rgb (in place), only within the face's bounding
    box, blending it in with a soft-edged mask."""
    height, width = rgb.shape[:2]
    inverse = cv2.invertAffineTransform(affine)
    corners = np.array([[0, 0, 1], [FACE_SIZE, 0, 1], [0, FACE_SIZE, 1], [FACE_SIZE, FACE_SIZE, 1]], dtype=np.float64) @ inverse.T
    x0, y0 = np.floor(corners.min(axis=0)).astype(int)
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return

    # Same transform, but into the coordinate system of the bounding box.
    local = inverse.copy()
    local[:, 2] -= (x0, y0)
    size = (x1 - x0, y1 - y0)
    face = cv2.warpAffine(restored, local, size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(np.ones((FACE_SIZE, FACE_SIZE), np.float32), local, size)

    # Feather: shrink the mask and blur its edge, scaled to the face size.
    edge = max(1, int(np.sqrt(mask.sum())) // 20)
    mask = cv2.erode(mask, np.ones((2 * edge + 1, 2 * edge + 1), np.uint8))
    mask = cv2.GaussianBlur(mask, (4 * edge + 1, 4 * edge + 1), 0)[..., None]

    region = rgb[y0:y1, x0:x1].astype(np.float32)
    rgb[y0:y1, x0:x1] = np.clip(mask * face.astype(np.float32) + (1 - mask) * region, 0, 255).astype(np.uint8)


def restore_faces(
    rgb: np.ndarray,
    detector: FaceDetector,
    restore: Callable[[np.ndarray], np.ndarray],
    strength: float = 1.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Restores all detected faces of rgb ((H, W, 3) uint8). restore maps an aligned
    512×512 uint8 face to a restored one; strength blends between the original (0) and
    the restored face (1). Raises NoFacesFoundError."""
    faces = [lm for lm in detector(rgb) if np.linalg.norm(lm[1] - lm[0]) >= MIN_EYE_DISTANCE]
    if not faces:
        raise NoFacesFoundError()

    result = rgb.copy()
    for done, landmarks in enumerate(faces, start=1):
        face, affine = align_face(rgb, landmarks)
        restored = restore(face).astype(np.float32)
        blended = np.clip(strength * restored + (1 - strength) * face.astype(np.float32), 0, 255).astype(np.uint8)
        paste_face(result, blended, affine)
        if on_progress:
            on_progress(done, len(faces))
    return result
