"""Classic (model-free) photo adjustments: exposure, white balance, contrast,
highlights/shadows, saturation/vibrance and sharpening.

All operations run as torch tensor ops, so they use the GPU when one is available
and fall back to the CPU otherwise (also on CUDA out-of-memory for very large photos).
Images are handled as float32 tensors of shape (3, H, W) with sRGB values in [0, 1].
"""

import io
import math
import os
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from fotoviewer.services.image_files import ImageNotFoundError, resolve_image_path
from fotoviewer.services.working_images import WorkingImages

PREVIEW_SIZE = 1600
EDITED_SUFFIX = "_edited"

# Rec. 709 luma weights (linear-light and sRGB-encoded luminance alike, close enough here).
_LUMA = (0.2126, 0.7152, 0.0722)
# Long side of the downscaled image used to compute the local brightness (base layer).
_BASE_LAYER_SIZE = 512
# Log-gain per slider unit for white balance (see _white_balance_gains).
_TEMPERATURE_SCALE = 0.5
_TINT_SCALE = 0.25


@dataclass
class Adjustments:
    """Slider values; 0 means "unchanged" for every field."""

    exposure: float = 0.0  # stops, -2 .. 2
    contrast: float = 0.0  # -1 .. 1
    highlights: float = 0.0  # -1 .. 1, negative recovers bright areas
    shadows: float = 0.0  # -1 .. 1, positive brightens dark areas
    temperature: float = 0.0  # -1 (cooler) .. 1 (warmer)
    tint: float = 0.0  # -1 (green) .. 1 (magenta)
    saturation: float = 0.0  # -1 .. 1
    vibrance: float = 0.0  # -1 .. 1, boosts muted colors more than saturated ones
    sharpen: float = 0.0  # 0 .. 1

    def is_identity(self) -> bool:
        return all(getattr(self, f.name) == 0 for f in fields(self))

    def to_dict(self) -> dict:
        return asdict(self)


# --- color space helpers -------------------------------------------------------------


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)


def _luma(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb[0], rgb[1], rgb[2]
    return _LUMA[0] * r + _LUMA[1] * g + _LUMA[2] * b


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - edge0) / (edge1 - edge0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


# --- filters -------------------------------------------------------------------------


def _box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Mean over a (2r+1)² window of a (H, W) tensor, borders handled by renormalizing."""
    kernel = 2 * radius + 1
    return F.avg_pool2d(x[None, None], kernel, stride=1, padding=radius, count_include_pad=False)[0, 0]


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (H, W) tensor."""
    radius = max(1, int(math.ceil(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    x = x[None, None]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1))
    x = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1))
    return x[0, 0]


def _base_layer(luma: torch.Tensor) -> torch.Tensor:
    """Edge-aware local brightness of the image (fast guided filter, He & Sun 2015).

    The filter coefficients are computed on a downscaled copy and applied at full
    resolution, so edges stay sharp (no halos) while the cost stays independent of
    the photo size."""
    h, w = luma.shape
    scale = min(1.0, _BASE_LAYER_SIZE / max(h, w))
    small = F.interpolate(luma[None, None], scale_factor=scale, mode="area")[0, 0] if scale < 1 else luma

    radius = max(2, round(max(small.shape) / 32))
    eps = 0.01
    mean_i = _box_filter(small, radius)
    var_i = _box_filter(small * small, radius) - mean_i * mean_i
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    if scale < 1:
        mean_a = F.interpolate(mean_a[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        mean_b = F.interpolate(mean_b[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return (mean_a * luma + mean_b).clamp(0, 1)


# --- the adjustment pipeline ---------------------------------------------------------


def _white_balance_gains(temperature: float, tint: float) -> tuple[float, float, float]:
    # Gains in linear light; at ±1 the red/blue ratio changes by a factor of e (~2.7).
    t = _TEMPERATURE_SCALE * temperature
    m = _TINT_SCALE * tint
    return math.exp(t), math.exp(-m), math.exp(-t)


def apply_adjustments(rgb: torch.Tensor, adj: Adjustments) -> torch.Tensor:
    """Applies all adjustments to a (3, H, W) sRGB tensor in [0, 1]."""
    if adj.is_identity():
        return rgb

    # 1. White balance and exposure in linear light, where they are physically meaningful.
    if adj.temperature or adj.tint or adj.exposure:
        linear = _srgb_to_linear(rgb)
        gains = torch.tensor(_white_balance_gains(adj.temperature, adj.tint), device=rgb.device, dtype=rgb.dtype)
        linear = linear * gains.view(3, 1, 1) * (2.0**adj.exposure)
        rgb = _linear_to_srgb(linear)
        # Soft-clip instead of hard-clipping bright areas that were pushed past white.
        rgb = torch.where(rgb > 0.9, 0.9 + 0.1 * torch.tanh((rgb - 0.9) / 0.1), rgb)

    # 2. Highlights / shadows: shift the local brightness, keep local detail and color.
    if adj.highlights or adj.shadows:
        luma = _luma(rgb)
        base = _base_layer(luma)
        shadow_weight = _smoothstep(0.55, 0.0, base)
        highlight_weight = _smoothstep(0.45, 1.0, base)
        delta = 0.45 * adj.shadows * shadow_weight * (1 - base) + 0.35 * adj.highlights * highlight_weight * base
        # Mix of multiplicative (keeps saturation) and additive shift (no noise blow-up in deep shadows).
        rgb = rgb + delta * (rgb + 0.05) / (luma + 0.05)

    rgb = rgb.clamp(0, 1)

    # 3. Contrast: S-curve around mid gray.
    if adj.contrast > 0:
        rgb = torch.lerp(rgb, rgb * rgb * (3 - 2 * rgb), adj.contrast)
    elif adj.contrast < 0:
        rgb = torch.lerp(rgb, 0.5 + (rgb - 0.5) * 0.5, -adj.contrast)

    # 4. Saturation and vibrance around the per-pixel luminance.
    if adj.saturation or adj.vibrance:
        luma = _luma(rgb)
        chroma = rgb.amax(0) - rgb.amin(0)
        factor = 1 + adj.saturation + adj.vibrance * (1 - chroma).clamp(0, 1) ** 2
        rgb = luma + (rgb - luma) * factor

    # 5. Sharpening: unsharp mask on luminance only, to avoid color fringes.
    if adj.sharpen > 0:
        luma = _luma(rgb.clamp(0, 1))
        sigma = max(1.0, max(rgb.shape[1:]) / 2500)
        detail = luma - _gaussian_blur(luma, sigma)
        rgb = rgb + 1.5 * adj.sharpen * detail

    return rgb.clamp(0, 1)


# --- automatic suggestion ------------------------------------------------------------


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def suggest_adjustments(rgb: torch.Tensor) -> Adjustments:
    """Derives slider values from image statistics (gray-world white balance, exposure
    towards a mid-gray median, highlight/shadow recovery, contrast from the tonal range).
    Corrections are deliberately damped, so scenes with an intended color cast or mood
    (sunsets, night shots) are not "fixed" all the way to neutral."""
    small = F.interpolate(rgb[None], size=(256, 256), mode="area")[0]
    linear = _srgb_to_linear(small)

    # Gray world on mid-tones only: ignore near-black and clipped pixels.
    luma_lin = _luma(linear)
    mask = (luma_lin > 0.02) & (luma_lin < 0.9)
    if mask.sum() < 100:
        mask = torch.ones_like(mask)
    means = torch.stack([linear[c][mask].mean() for c in range(3)]).clamp(min=1e-4)
    r_gain, g_gain, b_gain = (means.mean() / means).tolist()
    temperature = _clamp(0.6 * math.log(r_gain / b_gain) / (2 * _TEMPERATURE_SCALE))
    tint = _clamp(-0.6 * math.log(g_gain / math.sqrt(r_gain * b_gain)) / _TINT_SCALE)

    # Exposure: brighten underexposed photos towards a mid-gray key (geometric mean), but
    # only as far as the brightest areas allow — a dark subject (black sweater) next to a
    # well-exposed face must not blow out the face. What cannot be done via exposure is
    # left to the shadows slider. Bright scenes (snow, clouds) are never darkened.
    key = float(torch.exp(torch.log(luma_lin + 1e-4).mean()))
    desired = _clamp(0.6 * math.log2(0.18 / key), 0.0, 1.5)
    headroom = max(0.0, math.log2(0.95 / max(float(torch.quantile(luma_lin.flatten(), 0.995)), 1e-4)))
    exposure = min(desired, headroom)
    if exposure < 0.1:
        exposure = 0.0

    srgb_luma = _luma(_linear_to_srgb(linear * (2.0**exposure)).clamp(0, 1))
    clipped = float((srgb_luma > 0.97).float().mean())
    dark = float((srgb_luma < 0.12).float().mean())
    highlights = _clamp(-4 * clipped, -0.8, 0.0)
    shadows = _clamp(2 * dark + 0.3 * (desired - exposure), 0.0, 0.45)

    low, high = torch.quantile(srgb_luma.flatten(), torch.tensor([0.02, 0.98], device=rgb.device)).tolist()
    contrast = _clamp(0.9 - (high - low), 0.0, 0.4)

    return Adjustments(
        exposure=round(exposure, 2),
        contrast=round(contrast, 2),
        highlights=round(highlights, 2),
        shadows=round(shadows, 2),
        temperature=round(temperature, 2),
        tint=round(tint, 2),
        saturation=0.0,
        vibrance=0.15,
        sharpen=0.0,
    )


# --- file handling -------------------------------------------------------------------


def _pil_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(array).to(device).permute(2, 0, 1).float().div_(255)


def _tensor_to_pil(rgb: torch.Tensor) -> Image.Image:
    array = rgb.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, "RGB")


def edited_copy_path(full_path: Path) -> Path:
    """<name>_edited.<ext> next to the original, numbered if it already exists. BMP is
    written as PNG."""
    suffix = ".png" if full_path.suffix.lower() == ".bmp" else full_path.suffix
    candidate = full_path.with_name(f"{full_path.stem}{EDITED_SUFFIX}{suffix}")
    counter = 2
    while candidate.exists():
        candidate = full_path.with_name(f"{full_path.stem}{EDITED_SUFFIX}-{counter}{suffix}")
        counter += 1
    return candidate


class PhotoEnhancer:
    """Applies Adjustments to photos of the library on the given torch device.

    Every method optionally takes the id of an intermediate image produced by AI
    restoration steps (see WorkingImages); the adjustments are then applied to that image
    instead of the original, while EXIF data and file name still come from the original."""

    def __init__(self, device: str = "cpu", working_images: WorkingImages | None = None, preview_cache_size: int = 4) -> None:
        self.device = device
        self.working_images = working_images
        self._preview_cache: OrderedDict[tuple, Image.Image] = OrderedDict()
        self._preview_cache_size = preview_cache_size
        self._lock = threading.Lock()

    def _run(self, image: Image.Image, fn):
        """Runs fn(tensor) on the configured device, retrying on the CPU if the GPU runs
        out of memory (e.g. very large panoramas)."""
        try:
            return fn(_pil_to_tensor(image, self.device))
        except torch.OutOfMemoryError:
            if self.device == "cpu":
                raise
            torch.cuda.empty_cache()
            return fn(_pil_to_tensor(image, "cpu"))

    def source_path(self, images_path: str, relative_path: str, working_id: str | None = None) -> Path:
        """Path of the pixels to edit: the original or an intermediate image derived from
        it. Raises ImageNotFoundError."""
        full_path = resolve_image_path(images_path, relative_path)
        if not working_id:
            return full_path
        if self.working_images is None:
            raise ImageNotFoundError(working_id)
        return self.working_images.get(working_id, relative_path)[0]

    def _load_preview_source(self, images_path: str, relative_path: str, working_id: str | None = None) -> Image.Image:
        full_path = self.source_path(images_path, relative_path, working_id)
        stat = full_path.stat()
        key = (str(full_path), stat.st_mtime_ns, stat.st_size)
        with self._lock:
            cached = self._preview_cache.get(key)
            if cached is not None:
                self._preview_cache.move_to_end(key)
                return cached

        with Image.open(full_path) as image:
            # draft() lets the JPEG decoder downscale while decoding — much faster for large photos.
            image.draft("RGB", (PREVIEW_SIZE, PREVIEW_SIZE))
            image = ImageOps.exif_transpose(image)
            image.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE))
            image = image.convert("RGB")

        with self._lock:
            self._preview_cache[key] = image
            while len(self._preview_cache) > self._preview_cache_size:
                self._preview_cache.popitem(last=False)
        return image

    def preview_jpeg(
        self, images_path: str, relative_path: str, adjustments: Adjustments, working_id: str | None = None
    ) -> bytes:
        """Adjusted, downscaled preview (longest side <= PREVIEW_SIZE) as JPEG bytes.
        Raises ImageNotFoundError."""
        source = self._load_preview_source(images_path, relative_path, working_id)
        result = self._run(source, lambda rgb: _tensor_to_pil(apply_adjustments(rgb, adjustments)))
        buffer = io.BytesIO()
        result.save(buffer, "JPEG", quality=90)
        return buffer.getvalue()

    def suggest(self, images_path: str, relative_path: str, working_id: str | None = None) -> Adjustments:
        """Automatic slider values for a photo. Raises ImageNotFoundError."""
        source = self._load_preview_source(images_path, relative_path, working_id)
        return self._run(source, suggest_adjustments)

    def save_copy(
        self, images_path: str, relative_path: str, adjustments: Adjustments, working_id: str | None = None
    ) -> str:
        """Applies the adjustments at full resolution and writes them as a new file next
        to the original (which is never modified). EXIF data and the color profile are
        kept; the orientation is baked into the pixels. Returns the relative path of the
        new file. Raises ImageNotFoundError."""
        full_path = resolve_image_path(images_path, relative_path)
        pixel_path = self.source_path(images_path, relative_path, working_id)
        target = edited_copy_path(full_path)

        with Image.open(full_path) as original:
            exif = original.getexif()
            icc_profile = original.info.get("icc_profile")
            with Image.open(pixel_path) as source:
                image = ImageOps.exif_transpose(source)
                result = self._run(image, lambda rgb: _tensor_to_pil(apply_adjustments(rgb, adjustments)))

        exif[0x0112] = 1  # orientation: pixels are already upright
        save_kwargs = {"exif": exif.tobytes()}
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile
        if target.suffix.lower() in (".jpg", ".jpeg"):
            save_kwargs.update(quality=95, subsampling=0)
            image_format = "JPEG"
        else:
            image_format = "PNG"

        tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            result.save(tmp_path, image_format, **save_kwargs)
            os.replace(tmp_path, target)
        finally:
            tmp_path.unlink(missing_ok=True)
        return str(target.relative_to(Path(images_path).resolve()))
