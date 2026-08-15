from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response
from PIL import Image

from fotoviewer.api.dependencies import get_image_processor_from_request, get_individuals_index_from_request, get_settings
from fotoviewer.api.errors import not_found_as_http_404
from fotoviewer.core.config import Settings
from fotoviewer.schemas.identity import RegisterIndividualRequest, RenameIndividualRequest
from fotoviewer.services import individuals_store
from fotoviewer.services.identity import INDIVIDUALS_COLLECTION, crop_detection, register_individual
from fotoviewer.services.image_files import resolve_image_path
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.individuals_store import IndividualEntryNotFoundError

router = APIRouter()


@router.post("/api/individuals", status_code=204)
def create_individual(
    body: RegisterIndividualRequest,
    settings: Settings = Depends(get_settings),
    image_processor: ImageProcessor = Depends(get_image_processor_from_request),
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    with not_found_as_http_404("Image not found"):
        full_path = resolve_image_path(settings.images_path, body.path)

    with Image.open(full_path) as image:
        register_individual(image_processor, individuals_index, image.convert("RGB"), body.bbox, body.name)

    individuals_index.save(INDIVIDUALS_COLLECTION, str(Path(settings.index_dir) / "individuals.npz"))
    individuals_store.append_entry(settings.index_dir, individuals_index, body.path, body.bbox)


@router.get("/api/individuals/{entry_id}/thumbnail")
def individual_thumbnail(
    entry_id: str,
    settings: Settings = Depends(get_settings),
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    try:
        entry = individuals_store.get_entry(settings.index_dir, individuals_index, entry_id)
    except IndividualEntryNotFoundError:
        raise HTTPException(status_code=404, detail="Entry not found")

    if not entry["path"] or not entry["bbox"]:
        raise HTTPException(status_code=404, detail="No thumbnail available for this entry")

    with not_found_as_http_404("Source image not found"):
        full_path = resolve_image_path(settings.images_path, entry["path"])

    with Image.open(full_path) as image:
        crop = crop_detection(image.convert("RGB"), entry["bbox"])
        buf = BytesIO()
        crop.save(buf, format="JPEG")

    return Response(content=buf.getvalue(), media_type="image/jpeg")


@router.patch("/api/individuals/{entry_id}", status_code=204)
def rename_individual_entry(
    entry_id: str,
    body: RenameIndividualRequest,
    settings: Settings = Depends(get_settings),
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    try:
        individuals_store.rename_entry(settings.index_dir, individuals_index, entry_id, body.name)
    except IndividualEntryNotFoundError:
        raise HTTPException(status_code=404, detail="Entry not found")


@router.delete("/api/individuals/{entry_id}", status_code=204)
def delete_individual_entry(
    entry_id: str,
    settings: Settings = Depends(get_settings),
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    try:
        individuals_store.delete_entry(settings.index_dir, individuals_index, entry_id)
    except IndividualEntryNotFoundError:
        raise HTTPException(status_code=404, detail="Entry not found")


@router.patch("/api/individuals/groups/{name}", status_code=204)
def rename_individual_group(
    name: str,
    body: RenameIndividualRequest,
    settings: Settings = Depends(get_settings),
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    individuals_store.rename_group(settings.index_dir, individuals_index, name, body.name)


@router.delete("/api/individuals/groups/{name}", status_code=204)
def delete_individual_group(
    name: str,
    settings: Settings = Depends(get_settings),
    individuals_index: Index = Depends(get_individuals_index_from_request),
):
    individuals_store.delete_group(settings.index_dir, individuals_index, name)
