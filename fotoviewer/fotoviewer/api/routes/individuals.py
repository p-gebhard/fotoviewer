from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from PIL import Image

from fotoviewer.api.dependencies import get_image_processor_from_request, get_individuals_index_from_request
from fotoviewer.schemas.identity import RegisterIndividualRequest, RenameIndividualRequest
from fotoviewer.services import individuals_store
from fotoviewer.services.identity import INDIVIDUALS_COLLECTION, crop_detection, register_individual
from fotoviewer.services.image_files import ImageNotFoundError, resolve_image_path
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.individuals_store import IndividualEntryNotFoundError

router = APIRouter()


@router.post("/api/individuals", status_code=204)
def create_individual(
    body: RegisterIndividualRequest,
    request: Request,
    image_processor: ImageProcessor = Depends(get_image_processor_from_request),
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    settings = request.app.state.settings
    try:
        full_path = resolve_image_path(settings.images_path, body.path)
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")

    with Image.open(full_path) as image:
        register_individual(image_processor, individuals_index, image.convert("RGB"), body.bbox, body.name)

    individuals_index.save(INDIVIDUALS_COLLECTION, str(Path(settings.index_dir) / "individuals.npz"))
    individuals_store.append_entry(settings.index_dir, individuals_index, body.path, body.bbox)


@router.get("/api/individuals/{entry_id}/thumbnail")
def individual_thumbnail(
    entry_id: str,
    request: Request,
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    settings = request.app.state.settings
    try:
        entry = individuals_store.get_entry(settings.index_dir, individuals_index, entry_id)
    except IndividualEntryNotFoundError:
        raise HTTPException(status_code=404, detail="Entry not found")

    if not entry["path"] or not entry["bbox"]:
        raise HTTPException(status_code=404, detail="No thumbnail available for this entry")

    try:
        full_path = resolve_image_path(settings.images_path, entry["path"])
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail="Source image not found")

    with Image.open(full_path) as image:
        crop = crop_detection(image.convert("RGB"), entry["bbox"])
        buf = BytesIO()
        crop.save(buf, format="JPEG")

    return Response(content=buf.getvalue(), media_type="image/jpeg")


@router.patch("/api/individuals/{entry_id}", status_code=204)
def rename_individual_entry(
    entry_id: str,
    body: RenameIndividualRequest,
    request: Request,
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    settings = request.app.state.settings
    try:
        individuals_store.rename_entry(settings.index_dir, individuals_index, entry_id, body.name)
    except IndividualEntryNotFoundError:
        raise HTTPException(status_code=404, detail="Entry not found")


@router.delete("/api/individuals/{entry_id}", status_code=204)
def delete_individual_entry(
    entry_id: str,
    request: Request,
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    settings = request.app.state.settings
    try:
        individuals_store.delete_entry(settings.index_dir, individuals_index, entry_id)
    except IndividualEntryNotFoundError:
        raise HTTPException(status_code=404, detail="Entry not found")


@router.patch("/api/individuals/groups/{name}", status_code=204)
def rename_individual_group(
    name: str,
    body: RenameIndividualRequest,
    request: Request,
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    settings = request.app.state.settings
    individuals_store.rename_group(settings.index_dir, individuals_index, name, body.name)


@router.delete("/api/individuals/groups/{name}", status_code=204)
def delete_individual_group(
    name: str,
    request: Request,
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    settings = request.app.state.settings
    individuals_store.delete_group(settings.index_dir, individuals_index, name)
