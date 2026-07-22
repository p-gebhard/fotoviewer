from pydantic import BaseModel


class RegisterIndividualRequest(BaseModel):
    path: str
    bbox: list[float]
    name: str


class RenameIndividualRequest(BaseModel):
    name: str
