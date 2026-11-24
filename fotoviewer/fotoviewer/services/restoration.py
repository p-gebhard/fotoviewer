"""AI restoration tools (denoise, deblur, JPEG artifact removal, low-light enhancement,
retouching, upscaling, face restoration, object removal) based on pretrained models
loaded via spandrel.

Models are downloaded on first use (pinned by SHA256), loaded onto the GPU only for the
duration of one job and released afterwards, so they do not permanently compete with the
search and detection models for GPU memory. All models run in float32: half precision
produces NaNs on GTX 16xx cards and is not faster there.
"""

import hashlib
import math
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from fotoviewer.services.faces import FaceDetector, NoFacesFoundError, restore_faces, yunet_detector
from fotoviewer.services.inpainting import EmptyMaskError, inpaint

# Upscaling is refused when the result would exceed this size.
MAX_OUTPUT_MEGAPIXELS = 100
# Long side of the downscaled copy that "global" models (see RestorationTool.mode) run on.
GLOBAL_MAX_SIDE = 1024

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class WeightFile:
    filename: str
    url: str
    sha256: str


@dataclass(frozen=True)
class RestorationTool:
    key: str
    filename: str
    url: str
    sha256: str
    size_mb: int
    license: str
    # "tiled": the model runs on overlapping tiles of the full-resolution photo.
    # "global": the model needs the whole image (global illumination estimate); it runs on a
    #   downscaled copy and the correction is transferred to full resolution via a guided filter.
    # "faces": restores detected faces (needs a face detector as extra weights).
    # "inpaint": fills a user-painted mask (object removal).
    mode: Literal["tiled", "global", "faces", "inpaint"] = "tiled"
    scale: int = 1
    # Additional files the tool needs (e.g. the face detector).
    extra_weights: tuple[WeightFile, ...] = ()


TOOLS: dict[str, RestorationTool] = {
    tool.key: tool
    for tool in [
        RestorationTool(
            key="denoise",
            filename="scunet_color_real_psnr.pth",
            url="https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
            sha256="fa78899ba2caec9d235a900e91d96c689da71c42029230c2028b00f09f809c2e",
            size_mb=72,
            license="Apache-2.0 (SCUNet)",
        ),
        RestorationTool(
            key="deblur",
            filename="NAFNet-GoPro-width64.pth",
            url="https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width64.pth",
            sha256="329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f",
            size_mb=272,
            license="MIT (NAFNet)",
        ),
        RestorationTool(
            key="jpeg_artifacts",
            filename="fbcnn_color.pth",
            url="https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth",
            sha256="8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8",
            size_mb=288,
            license="Apache-2.0 (FBCNN)",
        ),
        RestorationTool(
            key="low_light",
            filename="Retinexformer_LOL_v2_real.pth",
            url="https://huggingface.co/CaiYuanhao/Retinexformer/resolve/main/LOL_v2_real.pth",
            sha256="539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b",
            size_mb=7,
            license="MIT (Retinexformer)",
            mode="global",
        ),
        RestorationTool(
            key="retouch",
            filename="Retinexformer_FiveK.pth",
            url="https://huggingface.co/CaiYuanhao/Retinexformer/resolve/main/FiveK.pth",
            sha256="800f6a9281fe8d95daca3108f2b826d5a2adead09031e0e998d30a615286d9c1",
            size_mb=7,
            license="MIT (Retinexformer)",
            mode="global",
        ),
        RestorationTool(
            key="upscale_2x",
            filename="RealESRGAN_x2plus.pth",
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
            size_mb=67,
            license="BSD-3-Clause (Real-ESRGAN)",
            scale=2,
        ),
        RestorationTool(
            key="upscale_4x",
            filename="RealESRGAN_x4plus.pth",
            url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
            size_mb=67,
            license="BSD-3-Clause (Real-ESRGAN)",
            scale=4,
        ),
        RestorationTool(
            key="faces",
            filename="GFPGANv1.4.pth",
            url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            sha256="e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
            size_mb=349,
            license="Apache-2.0 (GFPGAN), MIT (YuNet)",
            mode="faces",
            extra_weights=(
                WeightFile(
                    filename="face_detection_yunet_2023mar.onnx",
                    url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
                ),
            ),
        ),
        RestorationTool(
            key="inpaint",
            filename="big-lama.pt",
            url="https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt",
            sha256="344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9",
            size_mb=206,
            license="Apache-2.0 (LaMa)",
            mode="inpaint",
        ),
    ]
}


class OutputTooLargeError(ValueError):
    pass


@dataclass
class LoadedModel:
    """A model ready for inference: fn maps a (1, 3, h, w) tensor in [0, 1] to
    (1, 3, h*scale, w*scale). Inputs must be padded to a multiple of multiple_of and be
    at least minimum pixels on each side."""

    fn: Callable[[torch.Tensor], torch.Tensor]
    scale: int = 1
    multiple_of: int = 1
    minimum: int = 0
    release: Callable[[], None] = lambda: None


def load_spandrel_model(weights_path: Path, device: str) -> LoadedModel:
    from spandrel import ModelLoader  # imported lazily: only needed when a tool is used

    descriptor = ModelLoader().load_from_file(str(weights_path))
    descriptor.model.float()
    descriptor.to(torch.device(device)).eval()
    requirements = descriptor.size_requirements

    def release() -> None:
        descriptor.to(torch.device("cpu"))

    return LoadedModel(
        fn=descriptor,
        scale=descriptor.scale,
        multiple_of=requirements.multiple_of,
        minimum=requirements.minimum,
        release=release,
    )


def download_weights(tool: RestorationTool | WeightFile, target: Path, on_progress: ProgressCallback | None = None) -> None:
    """Downloads a weights file to target, verifying the SHA256 checksum."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.part")
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(tool.url, headers={"User-Agent": "fotoviewer"})
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp_path, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress("downloading", done, total)
        if digest.hexdigest() != tool.sha256:
            raise ValueError(f"Checksum mismatch for {tool.filename}")
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


# --- inference -----------------------------------------------------------------------


def _forward_padded(model: LoadedModel, x: torch.Tensor) -> torch.Tensor:
    """Runs the model on x (1, 3, h, w), padding to the model's size requirements."""
    h, w = x.shape[-2:]
    m = model.multiple_of
    target_h = max(math.ceil(h / m) * m, model.minimum)
    target_w = max(math.ceil(w / m) * m, model.minimum)
    if (target_h, target_w) != (h, w):
        x = F.pad(x, (0, target_w - w, 0, target_h - h), mode="replicate")
    with torch.inference_mode():
        y = model.fn(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    return y[..., : h * model.scale, : w * model.scale].clamp(0, 1)


def _run_tiled(
    model: LoadedModel,
    x: torch.Tensor,
    device: str,
    tile: int,
    overlap: int = 32,
    on_progress: ProgressCallback | None = None,
) -> torch.Tensor:
    """Processes x (1, 3, H, W, on the CPU) tile by tile; each tile is extended by
    `overlap` pixels of context on every side, which is cut off again afterwards, so there
    are no visible seams. The result is assembled on the CPU."""
    _, _, height, width = x.shape
    s = model.scale
    out = torch.empty((1, 3, height * s, width * s), dtype=torch.float32)
    rows = range(0, height, tile)
    cols = range(0, width, tile)
    total = len(rows) * len(cols)
    done = 0
    for y0 in rows:
        for x0 in cols:
            y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
            py0, px0 = max(0, y0 - overlap), max(0, x0 - overlap)
            py1, px1 = min(height, y1 + overlap), min(width, x1 + overlap)
            result = _forward_padded(model, x[..., py0:py1, px0:px1].to(device))
            oy, ox = (y0 - py0) * s, (x0 - px0) * s
            out[..., y0 * s : y1 * s, x0 * s : x1 * s] = result[
                ..., oy : oy + (y1 - y0) * s, ox : ox + (x1 - x0) * s
            ].cpu()
            done += 1
            if on_progress:
                on_progress("processing", done, total)
    return out


def _box(x: torch.Tensor, radius: int) -> torch.Tensor:
    return F.avg_pool2d(x, 2 * radius + 1, stride=1, padding=radius, count_include_pad=False)


def _guided_transfer(low_in: torch.Tensor, low_out: torch.Tensor, full_in: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Fits a local per-channel linear map low_in -> low_out (guided filter coefficients),
    upsamples it and applies it to the full-resolution input. This transfers the tonal and
    color correction of a model that only ran on a downscaled copy without losing detail."""
    eps = 1e-4
    mean_i = _box(low_in, radius)
    mean_o = _box(low_out, radius)
    cov = _box(low_in * low_out, radius) - mean_i * mean_o
    var = _box(low_in * low_in, radius) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_o - a * mean_i
    a, b = _box(a, radius), _box(b, radius)
    size = full_in.shape[-2:]
    a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
    b = F.interpolate(b, size=size, mode="bilinear", align_corners=False)
    return (a * full_in + b).clamp(0, 1)


def _run_global(model: LoadedModel, x: torch.Tensor, device: str, max_side: int, on_progress: ProgressCallback | None) -> torch.Tensor:
    _, _, height, width = x.shape
    if on_progress:
        on_progress("processing", 0, 1)
    factor = min(1.0, max_side / max(height, width))
    if factor < 1:
        low_in = F.interpolate(x, size=(max(1, round(height * factor)), max(1, round(width * factor))), mode="area")
    else:
        low_in = x
    low_out = _forward_padded(model, low_in.to(device)).cpu()
    result = low_out if factor == 1 else _guided_transfer(low_in, low_out, x)
    if on_progress:
        on_progress("processing", 1, 1)
    return result


def _is_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _to_uint8_image(x: torch.Tensor) -> np.ndarray:
    return x[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()


def _from_uint8_image(rgb: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(rgb)).to(device).permute(2, 0, 1)[None].float().div_(255)


def _run_faces(
    model: LoadedModel,
    rgb: np.ndarray,
    device: str,
    detector: FaceDetector,
    strength: float,
    on_progress: ProgressCallback | None,
) -> np.ndarray:
    def restore(face: np.ndarray) -> np.ndarray:
        return _to_uint8_image(_forward_padded(model, _from_uint8_image(face, device)))

    def progress(done: int, total: int) -> None:
        if on_progress:
            on_progress("processing", done, total)

    return restore_faces(rgb, detector, restore, strength, progress)


def _run_inpaint(model: LoadedModel, rgb: np.ndarray, mask: np.ndarray, device: str, on_progress: ProgressCallback | None) -> np.ndarray:
    def fill(crop: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
        h, w = crop_mask.shape
        m = model.multiple_of
        target_h, target_w = max(math.ceil(h / m) * m, model.minimum), max(math.ceil(w / m) * m, model.minimum)
        x = F.pad(_from_uint8_image(crop, device), (0, target_w - w, 0, target_h - h), mode="replicate")
        hole = torch.from_numpy(crop_mask.astype(np.float32)).to(device)[None, None]
        hole = F.pad(hole, (0, target_w - w, 0, target_h - h))
        with torch.inference_mode():
            y = model.fn(x, hole)
        return _to_uint8_image(y[..., :h, :w])

    if on_progress:
        on_progress("processing", 0, 1)
    result = inpaint(rgb, mask, fill)
    if on_progress:
        on_progress("processing", 1, 1)
    return result


class Restorer:
    """Runs RestorationTools on images; weights are cached in model_dir."""

    def __init__(
        self,
        device: str,
        model_dir: str,
        loader: Callable[[Path, str], LoadedModel] = load_spandrel_model,
        downloader: Callable[[RestorationTool | WeightFile, Path, ProgressCallback | None], None] = download_weights,
        tools: dict[str, RestorationTool] = TOOLS,
        face_detector_factory: Callable[[Path], FaceDetector] = yunet_detector,
    ) -> None:
        self.device = device
        self.model_dir = Path(model_dir)
        self.tools = tools
        self._loader = loader
        self._downloader = downloader
        self._face_detector_factory = face_detector_factory

    def weights_path(self, weights: RestorationTool | WeightFile) -> Path:
        return self.model_dir / weights.filename

    def is_downloaded(self, tool: RestorationTool) -> bool:
        return all(self.weights_path(w).is_file() for w in (tool, *tool.extra_weights))

    def tool_overview(self) -> list[dict]:
        return [
            {"key": t.key, "size_mb": t.size_mb, "scale": t.scale, "mode": t.mode, "downloaded": self.is_downloaded(t)}
            for t in self.tools.values()
        ]

    def check_output_size(self, tool: RestorationTool, width: int, height: int) -> None:
        if width * height * tool.scale**2 > MAX_OUTPUT_MEGAPIXELS * 1_000_000:
            raise OutputTooLargeError(tool.key)

    def run(
        self,
        tool_key: str,
        image: Image.Image,
        on_progress: ProgressCallback | None = None,
        mask: np.ndarray | None = None,
        strength: float = 1.0,
    ) -> Image.Image:
        """Applies a tool. mask ((H, W) bool, True = remove) is required for "inpaint";
        strength (0..1) blends restored faces with the originals for "faces"."""
        tool = self.tools[tool_key]
        self.check_output_size(tool, image.width, image.height)
        if tool.mode == "inpaint" and (mask is None or mask.shape != (image.height, image.width)):
            raise ValueError("inpaint needs a mask of the image size")

        for weights in (tool, *tool.extra_weights):
            if not self.weights_path(weights).is_file():
                self._downloader(weights, self.weights_path(weights), on_progress)

        if on_progress:
            on_progress("loading", 0, 0)
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        detector = self._face_detector_factory(self.weights_path(tool.extra_weights[0])) if tool.mode == "faces" else None

        # Try the configured device (with shrinking tiles for tiled models), finally the CPU.
        if self.device == "cpu":
            attempts = [("cpu", 512)]
        elif tool.mode in ("tiled", "global"):
            attempts = [(self.device, 512), (self.device, 256), (self.device, 128), ("cpu", 512)]
        else:
            attempts = [(self.device, 512), ("cpu", 512)]

        last_error: BaseException | None = None
        for device, tile in attempts:
            model = None
            try:
                model = self._loader(self.weights_path(tool), device)
                if tool.mode == "faces":
                    out = _run_faces(model, rgb, device, detector, strength, on_progress)
                elif tool.mode == "inpaint":
                    out = _run_inpaint(model, rgb, mask, device, on_progress)
                elif tool.mode == "global":
                    out = _to_uint8_image(_run_global(model, _from_uint8_image(rgb, "cpu"), device, GLOBAL_MAX_SIDE * tile // 512, on_progress))
                else:
                    out = _to_uint8_image(_run_tiled(model, _from_uint8_image(rgb, "cpu"), device, tile, on_progress=on_progress))
                break
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if not _is_out_of_memory(e):
                    raise
                last_error = e
            finally:
                if model is not None:
                    model.release()
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            raise last_error  # type: ignore[misc]

        return Image.fromarray(out, "RGB")


def run_restoration_job(
    restorer: Restorer,
    working_images,
    source_path: Path,
    relative_path: str,
    tool_key: str,
    previous_steps: list[str],
    status: dict,
    mask: Image.Image | None = None,
    strength: float = 1.0,
) -> None:
    """Background job: runs one tool on source_path (the original or an earlier
    intermediate result) and stores the output as a new intermediate image. Progress,
    result and errors are reported via status; known user-facing problems additionally
    set status["error_key"] ("no_faces", "empty_mask")."""

    def on_progress(phase: str, done: int, total: int) -> None:
        status["phase"] = phase
        status["progress"] = {"done": done, "total": total}

    try:
        with Image.open(source_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
        mask_array = None
        if mask is not None:
            # The mask was painted on a (possibly downscaled) preview; scale it to the photo.
            mask_array = np.array(mask.convert("L").resize(image.size, Image.Resampling.BILINEAR)) > 64
        result = restorer.run(tool_key, image, on_progress, mask=mask_array, strength=strength)
        status["phase"] = "storing"
        status.pop("progress", None)
        status["result"] = working_images.create(relative_path, [*previous_steps, tool_key], result)
        status["phase"] = "done"
    except NoFacesFoundError:
        status.update(error="no faces found", error_key="no_faces", phase="error")
    except EmptyMaskError:
        status.update(error="empty mask", error_key="empty_mask", phase="error")
    except Exception as e:
        status["error"] = str(e)
        status["phase"] = "error"
    finally:
        status["running"] = False
