from pydantic import BaseModel, field_validator


class ImportRequest(BaseModel):
    source_path: str
    destination_folder: str = ""
    new_folder_name: str = ""

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, source_path: str) -> str:
        source_path = source_path.strip()
        if not source_path:
            raise ValueError("source_path must not be empty")
        return source_path
