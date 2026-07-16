import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from fotoviewer.api.dependencies import (
    get_image_index,
    get_image_processor,
    get_individuals_index,
    get_object_detector,
)
from fotoviewer.schemas.websocket import (
    DetectObjectsInImageRequest,
    SearchImagesByTextRequest,
    SearchSimilarImagesRequest,
    WebSocketErrorResponse,
    WebSocketOkResponse,
    request_adapter,
)
from fotoviewer.services import commands
from fotoviewer.services.image_processor import ImageProcessor
from fotoviewer.services.indexing.index import Index
from fotoviewer.services.object_detector.object_detector import ObjectDetector

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/echo")
async def echo(
    websocket: WebSocket,
    image_processor: ImageProcessor = Depends(get_image_processor),
    object_detector: ObjectDetector = Depends(get_object_detector),
    image_index: Index = Depends(get_image_index),
    individuals_index: Index = Depends(get_individuals_index),
):
    identity_threshold = websocket.app.state.settings.identity_match_threshold
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                request = request_adapter.validate_json(raw)

                if isinstance(request, DetectObjectsInImageRequest):
                    result = await run_in_threadpool(
                        commands.detect_objects_in_image,
                        object_detector,
                        image_processor,
                        individuals_index,
                        request.image,
                        identity_threshold,
                    )
                elif isinstance(request, SearchSimilarImagesRequest):
                    result = await run_in_threadpool(
                        commands.search_similar_images, image_processor, image_index, request.image
                    )
                elif isinstance(request, SearchImagesByTextRequest):
                    result = await run_in_threadpool(
                        commands.search_images_by_text, image_processor, image_index, request.query
                    )

                response = WebSocketOkResponse(command=request.command, response=result)
            except ValidationError as e:
                response = WebSocketErrorResponse(reason=str(e))
            except Exception as e:
                logger.error(e)
                response = WebSocketErrorResponse(reason=f"Exception: {e}")

            await websocket.send_text(response.model_dump_json())
    except WebSocketDisconnect:
        pass
