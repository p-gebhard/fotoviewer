from pydantic import BaseModel, field_validator


class DeleteDuplicatesRequest(BaseModel):
    paths: list[str]

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, paths: list[str]) -> list[str]:
        if not paths:
            raise ValueError("paths must not be empty")
        return paths
