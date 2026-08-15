from fastapi import Depends, Request, WebSocket

from fotoviewer.api.errors import not_found_as_http_404
from fotoviewer.core.config import Settings
from fotoviewer.services.data_sources import get_data_overview
from fotoviewer.services.duplicates import find_duplicate_groups
from fotoviewer.services.folder_browser import list_folder
from fotoviewer.services.image_detection import detect_objects_in_file
from fotoviewer.services.image_metadata import get_image_metadata
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.detection_indexer import DETECTIONS_COLLECTION
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.individuals_store import list_individuals
from fotoviewer.services.library_filter import filter_library, list_folder_options
from fotoviewer.services.object_detector.object_detector import ObjectDetector


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_image_processor(websocket: WebSocket) -> ImageProcessor:
    return websocket.app.state.image_processor


def get_object_detector(websocket: WebSocket) -> ObjectDetector:
    return websocket.app.state.object_detector


def get_image_index(websocket: WebSocket) -> Index:
    return websocket.app.state.image_index


def get_individuals_index(websocket: WebSocket) -> Index:
    return websocket.app.state.individuals_index


def get_object_detector_from_request(request: Request) -> ObjectDetector:
    return request.app.state.object_detector


def get_image_processor_from_request(request: Request) -> ImageProcessor:
    return request.app.state.image_processor


def get_individuals_index_from_request(request: Request) -> Index:
    return request.app.state.individuals_index


def get_image_index_from_request(request: Request) -> Index:
    return request.app.state.image_index


def get_detections_index_from_request(request: Request) -> Index:
    return request.app.state.detections_index


def get_image_metadata_or_404(path: str, settings: Settings = Depends(get_settings)) -> dict:
    with not_found_as_http_404("Image not found"):
        return get_image_metadata(settings.images_path, path)


def get_image_detections_or_404(
    path: str,
    settings: Settings = Depends(get_settings),
    object_detector: ObjectDetector = Depends(get_object_detector_from_request),
    image_processor: ImageProcessor = Depends(get_image_processor_from_request),
    individuals_index: Index = Depends(get_individuals_index_from_request),
) -> dict:
    with not_found_as_http_404("Image not found"):
        return detect_objects_in_file(
            object_detector,
            image_processor,
            individuals_index,
            settings.images_path,
            path,
            settings.identity_match_threshold,
        )


def get_folder_listing_or_404(path: str = "", settings: Settings = Depends(get_settings)) -> dict:
    with not_found_as_http_404("Folder not found"):
        return list_folder(settings.images_path, path)


def get_folder_options_dependency(
    image_index: Index = Depends(get_image_index_from_request),
) -> list:
    return list_folder_options(image_index)


def get_data_overview_dependency(
    settings: Settings = Depends(get_settings),
    image_index: Index = Depends(get_image_index_from_request),
    individuals_index: Index = Depends(get_individuals_index_from_request),
    detections_index: Index = Depends(get_detections_index_from_request),
) -> dict:
    return get_data_overview(settings, image_index, individuals_index, detections_index)


def get_duplicate_groups_dependency(
    settings: Settings = Depends(get_settings),
    image_index: Index = Depends(get_image_index_from_request),
) -> list:
    return find_duplicate_groups(image_index, settings.duplicate_match_threshold, settings.images_path)


def get_individuals_overview_dependency(
    settings: Settings = Depends(get_settings),
    individuals_index: Index = Depends(get_individuals_index_from_request),
) -> list:
    return list_individuals(settings.index_dir, individuals_index)


def get_library_filter_dependency(
    folder: str = "",
    individual: str = "",
    year_from: str = "",
    year_to: str = "",
    settings: Settings = Depends(get_settings),
    image_index: Index = Depends(get_image_index_from_request),
    individuals_index: Index = Depends(get_individuals_index_from_request),
    detections_index: Index = Depends(get_detections_index_from_request),
) -> dict:
    parsed_year_from = int(year_from) if year_from.strip().isdigit() else None
    parsed_year_to = int(year_to) if year_to.strip().isdigit() else None

    files = filter_library(
        image_index,
        individuals_index,
        detections_index,
        settings.images_path,
        settings.identity_match_threshold,
        folder=folder or None,
        individual=individual or None,
        year_from=parsed_year_from,
        year_to=parsed_year_to,
    )

    return {
        "files": files,
        "folder_options": list_folder_options(image_index),
        "individual_options": [group["name"] for group in list_individuals(settings.index_dir, individuals_index)],
        "detections_indexed": detections_index.exists(DETECTIONS_COLLECTION)
        and detections_index.num_features(DETECTIONS_COLLECTION) > 0,
        "selected": {
            "folder": folder,
            "individual": individual,
            "year_from": year_from,
            "year_to": year_to,
        },
    }
