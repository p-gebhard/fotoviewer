import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from fotoviewer.services.restoration import (
    TOOLS,
    LoadedModel,
    OutputTooLargeError,
    Restorer,
    RestorationTool,
    _guided_transfer,
    _run_tiled,
    download_weights,
    run_restoration_job,
)
from fotoviewer.services.working_images import WorkingImages


def _image(width=300, height=200):
    image = Image.new("RGB", (width, height))
    image.putdata([((x * 7) % 256, (y * 5) % 256, (x + y) % 256) for y in range(height) for x in range(width)])
    return image


def _tensor(image):
    return torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1)[None].float() / 255


def test_tiled_identity_model_reproduces_input_exactly():
    x = _tensor(_image(301, 157))
    out = _run_tiled(LoadedModel(fn=lambda t: t), x, "cpu", tile=64, overlap=8)
    assert torch.equal(out, x)


def test_tiled_upscaling_assembles_full_output():
    x = _tensor(_image(130, 70))
    model = LoadedModel(fn=lambda t: torch.nn.functional.interpolate(t, scale_factor=2, mode="nearest"), scale=2)
    out = _run_tiled(model, x, "cpu", tile=48, overlap=8)
    assert torch.equal(out, torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


def test_tiled_pads_to_model_size_requirements():
    seen = []

    def fn(t):
        seen.append(tuple(t.shape[-2:]))
        return t

    x = _tensor(_image(50, 30))
    out = _run_tiled(LoadedModel(fn=fn, multiple_of=8, minimum=40), x, "cpu", tile=64, overlap=0)
    assert torch.equal(out, x)
    assert all(h % 8 == 0 and w % 8 == 0 and h >= 40 and w >= 40 for h, w in seen)


def test_tuple_outputs_are_unwrapped():
    x = _tensor(_image(40, 40))
    out = _run_tiled(LoadedModel(fn=lambda t: (t, "qf")), x, "cpu", tile=64)
    assert torch.equal(out, x)


def test_progress_reports_every_tile():
    calls = []
    _run_tiled(LoadedModel(fn=lambda t: t), _tensor(_image(100, 100)), "cpu", tile=50, on_progress=lambda *a: calls.append(a))
    assert calls == [("processing", i, 4) for i in range(1, 5)]


def test_guided_transfer_carries_global_correction_to_full_resolution():
    full = _tensor(_image(400, 300))
    low_in = torch.nn.functional.interpolate(full, size=(75, 100), mode="area")
    low_out = (low_in * 0.5 + 0.2).clamp(0, 1)  # a "model" that compresses contrast and brightens
    result = _guided_transfer(low_in, low_out, full)
    assert result.shape == full.shape
    assert torch.allclose(result, (full * 0.5 + 0.2).clamp(0, 1), atol=0.03)


def _restorer(tmp_path, loader, downloads=None):
    def downloader(tool, target, on_progress=None):
        if downloads is not None:
            downloads.append(getattr(tool, "key", tool.filename))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"w")

    return Restorer("cpu", str(tmp_path / "models"), loader=loader, downloader=downloader)


def test_run_downloads_once_and_applies_model(tmp_path):
    downloads = []
    restorer = _restorer(tmp_path, lambda path, device: LoadedModel(fn=lambda t: 1 - t), downloads)
    image = _image(80, 60)

    result = restorer.run("denoise", image)
    restorer.run("denoise", image)

    assert downloads == ["denoise"]
    assert result.size == image.size
    assert result.getpixel((10, 10)) == tuple(255 - v for v in image.getpixel((10, 10)))


def test_global_tools_run_on_downscaled_copy(tmp_path):
    sizes = []

    def loader(path, device):
        return LoadedModel(fn=lambda t: sizes.append(tuple(t.shape[-2:])) or t, multiple_of=8)

    restorer = _restorer(tmp_path, loader)
    result = restorer.run("retouch", _image(2400, 1200))

    assert result.size == (2400, 1200)
    assert max(sizes[0]) <= 1024 + 8


def test_run_rejects_too_large_output(tmp_path):
    restorer = _restorer(tmp_path, lambda path, device: LoadedModel(fn=lambda t: t))
    with pytest.raises(OutputTooLargeError):
        restorer.check_output_size(TOOLS["upscale_4x"], 4000, 3000)
    restorer.check_output_size(TOOLS["upscale_2x"], 4000, 3000)


def test_out_of_memory_falls_back_to_smaller_tiles(tmp_path):
    restorer = _restorer(tmp_path, None)
    restorer.device = "cpu:0"
    attempts = []

    def loader(path, device):
        attempts.append(device)

        def fn(t):
            if device != "cpu":
                raise torch.OutOfMemoryError("CUDA out of memory")
            return t

        return LoadedModel(fn=fn)

    restorer._loader = loader
    result = restorer.run("denoise", _image(40, 40))

    assert attempts == ["cpu:0", "cpu:0", "cpu:0", "cpu"]
    assert result.size == (40, 40)


def test_other_errors_are_not_swallowed(tmp_path):
    def loader(path, device):
        return LoadedModel(fn=lambda t: (_ for _ in ()).throw(RuntimeError("shape mismatch")))

    with pytest.raises(RuntimeError, match="shape mismatch"):
        _restorer(tmp_path, loader).run("denoise", _image(40, 40))


def test_download_verifies_checksum(tmp_path):
    payload = b"model weights"
    source = tmp_path / "source.pth"
    source.write_bytes(payload)
    good = RestorationTool("t", "t.pth", source.as_uri(), hashlib.sha256(payload).hexdigest(), 1, "MIT")
    bad = RestorationTool("t", "t.pth", source.as_uri(), "0" * 64, 1, "MIT")

    progress = []
    download_weights(good, tmp_path / "models" / "t.pth", lambda *a: progress.append(a))
    assert (tmp_path / "models" / "t.pth").read_bytes() == payload
    assert progress[-1][0] == "downloading"

    with pytest.raises(ValueError, match="Checksum"):
        download_weights(bad, tmp_path / "other" / "t.pth")
    assert not list((tmp_path / "other").iterdir())


def test_all_tools_have_pinned_checksums_and_https_urls():
    for tool in TOOLS.values():
        assert len(tool.sha256) == 64
        assert tool.url.startswith("https://")


def test_job_stores_result_as_working_image(tmp_path):
    restorer = _restorer(tmp_path, lambda path, device: LoadedModel(fn=lambda t: 1 - t))
    working = WorkingImages(str(tmp_path / "edits"))
    source = tmp_path / "a.jpg"
    _image(64, 48).save(source)
    status = {"running": True}

    run_restoration_job(restorer, working, source, "a.jpg", "denoise", ["retouch"], status)

    assert status["phase"] == "done" and status["running"] is False
    meta = status["result"]
    assert meta["steps"] == ["retouch", "denoise"]
    path, stored = working.get(meta["id"], "a.jpg")
    assert Image.open(path).size == (64, 48)


def test_job_reports_errors(tmp_path):
    restorer = _restorer(tmp_path, lambda path, device: LoadedModel(fn=lambda t: t))
    status = {"running": True}
    run_restoration_job(restorer, WorkingImages(str(tmp_path)), tmp_path / "missing.jpg", "x", "denoise", [], status)
    assert status["phase"] == "error" and status["error"] and status["running"] is False


def test_faces_tool_downloads_detector_and_restores(tmp_path):
    import numpy as np

    from fotoviewer.services.faces import FFHQ_TEMPLATE

    downloads = []
    restorer = _restorer(tmp_path, lambda path, device: LoadedModel(fn=lambda t: torch.ones_like(t), minimum=512), downloads)
    restorer._face_detector_factory = lambda path: (lambda rgb: [FFHQ_TEMPLATE * 0.5 + 20])
    image = Image.new("RGB", (400, 300), (10, 10, 10))

    result = restorer.run("faces", image, strength=1.0)

    assert downloads == ["faces", "face_detection_yunet_2023mar.onnx"]
    nose = (FFHQ_TEMPLATE[2] * 0.5 + 20).astype(int)
    assert result.getpixel(tuple(nose)) == (255, 255, 255)
    assert result.getpixel((390, 290)) == (10, 10, 10)


def test_inpaint_tool_passes_mask_to_model(tmp_path):
    import numpy as np

    seen = []

    def loader(path, device):
        def fn(x, mask):
            seen.append((tuple(x.shape), tuple(mask.shape), float(mask.max())))
            return torch.zeros_like(x)

        return LoadedModel(fn=fn, multiple_of=8)

    restorer = _restorer(tmp_path, loader)
    image = Image.new("RGB", (200, 100), (200, 200, 200))
    mask = np.zeros((100, 200), bool)
    mask[40:60, 90:110] = True

    result = restorer.run("inpaint", image, mask=mask)

    (x_shape, mask_shape, mask_max), = seen
    assert x_shape[-2:] == mask_shape[-2:] and x_shape[-1] % 8 == 0 and x_shape[-2] % 8 == 0
    assert mask_max == 1.0
    assert result.getpixel((100, 50)) == (0, 0, 0)
    assert result.getpixel((5, 5)) == (200, 200, 200)


def test_inpaint_requires_matching_mask(tmp_path):
    restorer = _restorer(tmp_path, lambda path, device: LoadedModel(fn=lambda x, m: x))
    with pytest.raises(ValueError):
        restorer.run("inpaint", _image(40, 40))


def test_job_reports_known_errors_with_key(tmp_path):
    restorer = _restorer(tmp_path, lambda path, device: LoadedModel(fn=lambda t: t))
    restorer._face_detector_factory = lambda path: (lambda rgb: [])
    source = tmp_path / "a.jpg"
    _image(64, 48).save(source)
    status = {"running": True}

    run_restoration_job(restorer, WorkingImages(str(tmp_path / "edits")), source, "a.jpg", "faces", [], status)

    assert status["phase"] == "error" and status["error_key"] == "no_faces"


def test_job_scales_mask_to_photo_size(tmp_path):
    import numpy as np

    seen = []
    restorer = _restorer(tmp_path, lambda path, device: LoadedModel(fn=lambda x, m: seen.append(tuple(m.shape)) or x))
    source = tmp_path / "a.png"
    _image(400, 200).save(source)
    mask = Image.new("L", (100, 50))
    mask.paste(255, (40, 20, 60, 30))
    status = {"running": True}

    run_restoration_job(restorer, WorkingImages(str(tmp_path / "edits")), source, "a.png", "inpaint", [], status, mask=mask)

    assert status["phase"] == "done", status
    assert seen  # model ran on a crop of the full-size photo
