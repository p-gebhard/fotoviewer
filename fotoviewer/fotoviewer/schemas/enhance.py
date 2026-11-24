from pydantic import BaseModel, Field

from fotoviewer.services.enhance import Adjustments


class AdjustmentValues(BaseModel):
    exposure: float = Field(0.0, ge=-2, le=2)
    contrast: float = Field(0.0, ge=-1, le=1)
    highlights: float = Field(0.0, ge=-1, le=1)
    shadows: float = Field(0.0, ge=-1, le=1)
    temperature: float = Field(0.0, ge=-1, le=1)
    tint: float = Field(0.0, ge=-1, le=1)
    saturation: float = Field(0.0, ge=-1, le=1)
    vibrance: float = Field(0.0, ge=-1, le=1)
    sharpen: float = Field(0.0, ge=0, le=1)

    def to_adjustments(self) -> Adjustments:
        return Adjustments(**self.model_dump())


class EnhanceRequest(BaseModel):
    path: str
    adjustments: AdjustmentValues = AdjustmentValues()
    # Intermediate result of AI restoration steps to edit instead of the original.
    working_id: str | None = None


class RestoreRequest(BaseModel):
    path: str
    tool: str
    working_id: str | None = None
    # Object removal: PNG data URL of the painted mask (any non-transparent pixel = remove).
    mask: str | None = Field(None, max_length=20_000_000)
    # Face restoration: blend between original (0) and restored face (1).
    strength: float = Field(0.7, ge=0, le=1)


class EnhanceSaveResponse(BaseModel):
    path: str
    reindex_started: bool
