import pytest
import torch
from PIL import Image

from fotoviewer.services.enhance import Adjustments, PhotoEnhancer, apply_adjustments, edited_copy_path, suggest_adjustments
from fotoviewer.services.image_files import ImageNotFoundError


def _gradient(height: int = 64, width: int = 96) -> torch.Tensor:
    ramp = torch.linspace(0, 1, width).expand(height, width)
    return torch.stack([ramp, ramp.flip(1), torch.full((height, width), 0.4)])


def _mean_luma(rgb: torch.Tensor) -> float:
    return float((0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).mean())


def test_identity_adjustments_return_input_unchanged():
    rgb = _gradient()
    assert torch.equal(apply_adjustments(rgb, Adjustments()), rgb)


@pytest.mark.parametrize(
    "field",
    ["exposure", "contrast", "highlights", "shadows", "temperature", "tint", "saturation", "vibrance", "sharpen"],
)
@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_every_adjustment_stays_in_range_and_keeps_shape(field, value):
    if field == "sharpen" and value < 0:
        pytest.skip("sharpen is 0 .. 1")
    rgb = _gradient()
    result = apply_adjustments(rgb, Adjustments(**{field: value}))
    assert result.shape == rgb.shape
    assert result.min() >= 0 and result.max() <= 1
    assert not torch.equal(result, rgb)


def test_exposure_brightens_and_darkens():
    rgb = torch.full((3, 32, 32), 0.3)
    assert _mean_luma(apply_adjustments(rgb, Adjustments(exposure=1))) > 0.35
    assert _mean_luma(apply_adjustments(rgb, Adjustments(exposure=-1))) < 0.25


def test_shadows_lift_dark_areas_more_than_bright_ones():
    rgb = torch.cat([torch.full((3, 32, 32), 0.1), torch.full((3, 32, 32), 0.9)], dim=2)
    result = apply_adjustments(rgb, Adjustments(shadows=1))
    dark_gain = float(result[:, :, :32].mean() - 0.1)
    bright_gain = float(result[:, :, 32:].mean() - 0.9)
    assert dark_gain > 0.05
    assert abs(bright_gain) < dark_gain / 3


def test_highlights_negative_darkens_bright_areas():
    rgb = torch.full((3, 32, 32), 0.95)
    assert _mean_luma(apply_adjustments(rgb, Adjustments(highlights=-1))) < 0.9


def test_temperature_shifts_red_blue_balance():
    gray = torch.full((3, 16, 16), 0.5)
    warm = apply_adjustments(gray, Adjustments(temperature=1))
    cool = apply_adjustments(gray, Adjustments(temperature=-1))
    assert warm[0].mean() > warm[2].mean()
    assert cool[2].mean() > cool[0].mean()


def test_saturation_minus_one_yields_grayscale():
    result = apply_adjustments(_gradient(), Adjustments(saturation=-1))
    assert torch.allclose(result[0], result[1], atol=1e-5)
    assert torch.allclose(result[1], result[2], atol=1e-5)


def test_suggest_neutralizes_color_cast_and_brightens_dark_photo():
    # Dark photo with a strong blue cast.
    rgb = torch.stack([torch.full((64, 64), 0.08), torch.full((64, 64), 0.1), torch.full((64, 64), 0.3)])
    rgb = rgb + 0.05 * torch.rand_like(rgb)
    suggestion = suggest_adjustments(rgb)
    assert suggestion.exposure > 0
    assert suggestion.temperature > 0  # warm up a blue image

    corrected = apply_adjustments(rgb, suggestion)
    assert _mean_luma(corrected) > _mean_luma(rgb)
    assert float(corrected[2].mean() / corrected[0].mean()) < 0.8 * float(rgb[2].mean() / rgb[0].mean())


def test_suggest_leaves_well_exposed_neutral_photo_mostly_alone():
    gray_ramp = torch.linspace(0.05, 0.95, 128).expand(3, 128, 128).clone()
    suggestion = suggest_adjustments(gray_ramp)
    assert abs(suggestion.temperature) < 0.3
    assert abs(suggestion.tint) < 0.1
    assert abs(suggestion.exposure) <= 0.5


def test_edited_copy_path_numbers_existing_copies(tmp_path):
    original = tmp_path / "photo.jpg"
    original.touch()
    assert edited_copy_path(original).name == "photo_edited.jpg"
    (tmp_path / "photo_edited.jpg").touch()
    assert edited_copy_path(original).name == "photo_edited-2.jpg"
    assert edited_copy_path(tmp_path / "scan.bmp").name == "scan_edited.png"


def test_save_copy_keeps_original_and_exif(tmp_path):
    (tmp_path / "sub").mkdir()
    image = Image.new("RGB", (300, 200), (90, 90, 140))
    exif = Image.Exif()
    exif[0x0110] = "TestCam"  # Model
    exif[0x0112] = 6  # rotated 90° clockwise
    image.save(tmp_path / "sub" / "a.jpg", exif=exif.tobytes())
    original_bytes = (tmp_path / "sub" / "a.jpg").read_bytes()

    new_path = PhotoEnhancer("cpu").save_copy(str(tmp_path), "sub/a.jpg", Adjustments(exposure=0.5))

    assert new_path == "sub/a_edited.jpg"
    assert (tmp_path / "sub" / "a.jpg").read_bytes() == original_bytes
    with Image.open(tmp_path / new_path) as saved:
        assert saved.size == (200, 300)  # orientation baked into the pixels
        saved_exif = saved.getexif()
        assert saved_exif[0x0110] == "TestCam"
        assert saved_exif[0x0112] == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_preview_is_downscaled_jpeg(tmp_path):
    Image.new("RGB", (4000, 1000), (120, 100, 80)).save(tmp_path / "wide.jpg")
    jpeg = PhotoEnhancer("cpu").preview_jpeg(str(tmp_path), "wide.jpg", Adjustments(contrast=0.5))
    preview = Image.open(__import__("io").BytesIO(jpeg))
    assert preview.format == "JPEG"
    assert max(preview.size) <= 1600


def test_preview_raises_for_missing_image(tmp_path):
    with pytest.raises(ImageNotFoundError):
        PhotoEnhancer("cpu").preview_jpeg(str(tmp_path), "missing.jpg", Adjustments())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_and_cpu_results_match():
    rgb = _gradient(128, 128)
    adj = Adjustments(exposure=0.4, shadows=0.5, highlights=-0.5, contrast=0.3, temperature=0.2, vibrance=0.3, sharpen=0.5)
    cpu = apply_adjustments(rgb, adj)
    gpu = apply_adjustments(rgb.cuda(), adj).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-3)
