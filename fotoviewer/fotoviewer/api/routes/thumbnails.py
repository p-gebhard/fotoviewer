from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from fotoviewer.api.dependencies import get_settings
from fotoviewer.api.errors import not_found_as_http_404
from fotoviewer.core.config import Settings
from fotoviewer.services.thumbnails import THUMBNAIL_DIRNAME, get_thumbnail

router = APIRouter()


@router.get("/thumbnails")
def thumbnail(path: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    try:
        with not_found_as_http_404("Image not found"):
            thumb_path = get_thumbnail(settings.images_path, str(Path(settings.index_dir) / THUMBNAIL_DIRNAME), path)
    except OSError:
        raise HTTPException(status_code=415, detail="Image could not be read")
    # no-cache = the browser revalidates via ETag, so a modified photo never shows a stale thumbnail.
    return FileResponse(thumb_path, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})
