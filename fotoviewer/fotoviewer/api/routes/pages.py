from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from fotoviewer.api.dependencies import get_image_metadata_or_404

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return request.app.state.templates.TemplateResponse(request, "index.html")


@router.get("/image", response_class=HTMLResponse)
def image_detail(request: Request, metadata: dict = Depends(get_image_metadata_or_404)):
    return request.app.state.templates.TemplateResponse(request, "image_detail.html", {"metadata": metadata})
