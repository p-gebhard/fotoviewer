from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from fotoviewer.api.dependencies import (
    get_data_overview_dependency,
    get_duplicate_groups_dependency,
    get_folder_listing_or_404,
    get_image_metadata_or_404,
)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return request.app.state.templates.TemplateResponse(request, "index.html", {"active_page": "search"})


@router.get("/image", response_class=HTMLResponse)
def image_detail(request: Request, metadata: dict = Depends(get_image_metadata_or_404)):
    return request.app.state.templates.TemplateResponse(request, "image_detail.html", {"metadata": metadata})


@router.get("/folders", response_class=HTMLResponse)
def folders(request: Request, listing: dict = Depends(get_folder_listing_or_404)):
    return request.app.state.templates.TemplateResponse(
        request, "folders.html", {"active_page": "folders", "listing": listing}
    )


@router.get("/sources", response_class=HTMLResponse)
def sources(request: Request, overview: dict = Depends(get_data_overview_dependency)):
    return request.app.state.templates.TemplateResponse(
        request, "sources.html", {"active_page": "sources", "overview": overview}
    )


@router.get("/duplicates", response_class=HTMLResponse)
def duplicates(request: Request, groups: list = Depends(get_duplicate_groups_dependency)):
    return request.app.state.templates.TemplateResponse(
        request, "duplicates.html", {"active_page": "duplicates", "groups": groups}
    )
