import base64
import binascii
import io
import threading

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from PIL import Image

from fotoviewer.api.dependencies import get_photo_enhancer_from_request, get_settings
from fotoviewer.api.errors import not_found_as_http_404
from fotoviewer.api.routes.reindex import start_background_reindex
from fotoviewer.core.config import Settings
from fotoviewer.schemas.enhance import AdjustmentValues, EnhanceRequest, EnhanceSaveResponse, RestoreRequest
from fotoviewer.services.enhance import PhotoEnhancer
from fotoviewer.services.restoration import OutputTooLargeError, run_restoration_job

router = APIRouter()


@router.post("/api/images/enhance/preview")
def enhance_preview(
    body: EnhanceRequest,
    settings: Settings = Depends(get_settings),
    enhancer: PhotoEnhancer = Depends(get_photo_enhancer_from_request),
) -> Response:
    try:
        with not_found_as_http_404("Image not found"):
            jpeg = enhancer.preview_jpeg(
                settings.images_path, body.path, body.adjustments.to_adjustments(), body.working_id
            )
    except OSError:
        raise HTTPException(status_code=415, detail="Image could not be read")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/api/images/enhance/auto", response_model=AdjustmentValues)
def enhance_auto(
    path: str,
    working_id: str | None = None,
    settings: Settings = Depends(get_settings),
    enhancer: PhotoEnhancer = Depends(get_photo_enhancer_from_request),
) -> AdjustmentValues:
    try:
        with not_found_as_http_404("Image not found"):
            return AdjustmentValues(**enhancer.suggest(settings.images_path, path, working_id).to_dict())
    except OSError:
        raise HTTPException(status_code=415, detail="Image could not be read")


@router.post("/api/images/enhance/save", response_model=EnhanceSaveResponse)
def enhance_save(
    body: EnhanceRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    enhancer: PhotoEnhancer = Depends(get_photo_enhancer_from_request),
) -> EnhanceSaveResponse:
    try:
        with not_found_as_http_404("Image not found"):
            new_path = enhancer.save_copy(
                settings.images_path, body.path, body.adjustments.to_adjustments(), body.working_id
            )
    except OSError:
        raise HTTPException(status_code=415, detail="Image could not be read")
    # Make the copy searchable; if an indexing run is already active, the next one picks it up.
    return EnhanceSaveResponse(path=new_path, reindex_started=start_background_reindex(request.app, settings))


def _decode_mask(data_url: str) -> Image.Image:
    """Decodes a PNG data URL into an "L" mask (255 = remove). Raises ValueError if it is
    invalid or empty."""
    header, _, payload = data_url.partition(",")
    if not header.startswith("data:image/png;base64"):
        raise ValueError("mask must be a PNG data URL")
    try:
        with Image.open(io.BytesIO(base64.b64decode(payload, validate=True))) as image:
            image.load()
            mask = image.getchannel("A") if "A" in image.getbands() else image.convert("L")
    except (binascii.Error, OSError) as e:
        raise ValueError("mask could not be decoded") from e
    mask = mask.point(lambda v: 255 if v > 0 else 0)
    if mask.getbbox() is None:
        raise ValueError("mask is empty")
    return mask


@router.post("/api/images/restore", status_code=202)
def start_restoration(
    body: RestoreRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    enhancer: PhotoEnhancer = Depends(get_photo_enhancer_from_request),
) -> dict:
    """Starts an AI restoration step in the background; poll /api/images/restore/status."""
    restorer = request.app.state.restorer
    translator = request.app.state.translator
    tool = restorer.tools.get(body.tool)
    if tool is None:
        raise HTTPException(status_code=422, detail=f"Unknown tool: {body.tool}")
    mask = None
    if tool.mode == "inpaint":
        try:
            mask = _decode_mask(body.mask or "")
        except ValueError:
            raise HTTPException(status_code=422, detail=translator("errors.empty_mask"))

    with not_found_as_http_404("Image not found"):
        source = enhancer.source_path(settings.images_path, body.path, body.working_id)
        steps = enhancer.working_images.get(body.working_id, body.path)[1]["steps"] if body.working_id else []

    try:
        with Image.open(source) as image:
            restorer.check_output_size(tool, image.width, image.height)
    except OutputTooLargeError:
        raise HTTPException(status_code=422, detail=translator("errors.output_too_large"))
    except OSError:
        raise HTTPException(status_code=415, detail="Image could not be read")

    status = request.app.state.restore_status
    with request.app.state.restore_lock:
        if status["running"]:
            raise HTTPException(status_code=409, detail=translator("errors.restoration_running"))
        status.clear()
        status.update(running=True, phase="starting", error=None, result=None, path=body.path, tool=body.tool)

    threading.Thread(
        target=run_restoration_job,
        args=(restorer, enhancer.working_images, source, body.path, body.tool, steps, status, mask, body.strength),
        daemon=True,
    ).start()
    return {"status": "started"}


@router.get("/api/images/restore/status")
def restoration_status(request: Request) -> dict:
    return request.app.state.restore_status
