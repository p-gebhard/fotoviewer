from pydantic import BaseModel, field_validator


def _validate_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("name must not be empty")
    return name


class RegisterIndividualRequest(BaseModel):
    path: str
    bbox: list[float]
    name: str

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, bbox: list[float]) -> list[float]:
        if len(bbox) != 4:
            raise ValueError("bbox must have exactly 4 values [x1, y1, x2, y2]")
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have x2 > x1 and y2 > y1")
        return bbox

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        return _validate_name(name)


class RenameIndividualRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        return _validate_name(name)
