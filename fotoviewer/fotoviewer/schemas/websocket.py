from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from fotoviewer.schemas.detection import Detection


class DetectObjectsInImageRequest(BaseModel):
    command: Literal["detect_objects_in_image"]
    image: str


class SearchSimilarImagesRequest(BaseModel):
    command: Literal["search_similar_images"]
    image: str


class SearchImagesByTextRequest(BaseModel):
    command: Literal["search_images_by_text"]
    query: str


WebSocketRequest = Annotated[
    Union[DetectObjectsInImageRequest, SearchSimilarImagesRequest, SearchImagesByTextRequest],
    Field(discriminator="command"),
]
request_adapter = TypeAdapter(WebSocketRequest)


class SearchResult(BaseModel):
    file: str
    score: float


class DetectObjectsResult(BaseModel):
    image: str
    detections: list[Detection]


class WebSocketOkResponse(BaseModel):
    status: Literal["ok"] = "ok"
    command: str
    response: DetectObjectsResult | list[SearchResult]


class WebSocketErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    reason: str
