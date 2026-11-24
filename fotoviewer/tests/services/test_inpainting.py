import numpy as np
import pytest

from fotoviewer.services import inpainting
from fotoviewer.services.inpainting import EmptyMaskError, inpaint


def _fill_with(value, calls=None):
    def model(crop, mask):
        if calls is not None:
            calls.append((crop.shape, mask.shape, mask.dtype))
        return np.full_like(crop, value)

    return model


def test_masked_area_is_filled_rest_untouched():
    photo = np.full((300, 400, 3), 50, np.uint8)
    mask = np.zeros((300, 400), bool)
    mask[100:150, 200:260] = True

    result = inpaint(photo, mask, _fill_with(200))

    assert (result[100:150, 200:260] == 200).all()
    assert (result[:50] == 50).all()  # far away from the mask
    assert (result[:, :100] == 50).all()


def test_only_a_crop_around_the_mask_is_processed():
    calls = []
    photo = np.zeros((1000, 1500, 3), np.uint8)
    mask = np.zeros((1000, 1500), bool)
    mask[500:520, 700:730] = True

    inpaint(photo, mask, _fill_with(1, calls))

    (crop_shape, mask_shape, dtype), = calls
    assert crop_shape[:2] == mask_shape
    assert dtype == bool
    assert crop_shape[0] < 400 and crop_shape[1] < 400


def test_large_crops_are_downscaled_for_the_model(monkeypatch):
    monkeypatch.setattr(inpainting, "MAX_INPAINT_SIDE", 128)
    calls = []
    photo = np.zeros((600, 800, 3), np.uint8)
    mask = np.zeros((600, 800), bool)
    mask[100:500, 100:700] = True

    result = inpaint(photo, mask, _fill_with(255, calls))

    assert max(calls[0][0][:2]) <= 128
    assert result.shape == photo.shape
    assert (result[300, 400] == 255).all()


def test_empty_mask_raises():
    with pytest.raises(EmptyMaskError):
        inpaint(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), bool), _fill_with(0))
