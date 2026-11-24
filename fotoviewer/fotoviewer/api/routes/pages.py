from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse

from fotoviewer.api.dependencies import (
    get_data_overview_dependency,
    get_duplicate_groups_dependency,
    get_folder_listing_or_404,
    get_folder_options_dependency,
    get_image_index_from_request,
    get_image_metadata_or_404,
    get_individuals_overview_dependency,
    get_library_filter_dependency,
)
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.restoration import MAX_OUTPUT_MEGAPIXELS

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index(request: Request, image_index: Index = Depends(get_image_index_from_request)):
    index_empty = not image_index.exists("image") or image_index.num_features("image") == 0
    return request.app.state.templates.TemplateResponse(
        request, "index.html", {"active_page": "search", "index_empty": index_empty}
    )


@router.get("/image", response_class=HTMLResponse)
def image_detail(request: Request, metadata: dict = Depends(get_image_metadata_or_404)):
    return request.app.state.templates.TemplateResponse(
        request,
        "image_detail.html",
        {
            "metadata": metadata,
            "restoration_tools": request.app.state.restorer.tool_overview(),
            "max_output_megapixels": MAX_OUTPUT_MEGAPIXELS,
        },
    )


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


@router.get("/individuals", response_class=HTMLResponse)
def individuals(request: Request, groups: list = Depends(get_individuals_overview_dependency)):
    return request.app.state.templates.TemplateResponse(
        request, "individuals.html", {"active_page": "individuals", "groups": groups}
    )


@router.get("/filter", response_class=HTMLResponse)
def library_filter(request: Request, result: dict = Depends(get_library_filter_dependency)):
    return request.app.state.templates.TemplateResponse(
        request, "filter.html", {"active_page": "filter", **result}
    )


@router.get("/import", response_class=HTMLResponse)
def import_page(request: Request, folder_options: list = Depends(get_folder_options_dependency)):
    return request.app.state.templates.TemplateResponse(
        request, "import.html", {"active_page": "import", "folder_options": folder_options}
    )
