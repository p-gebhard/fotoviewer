from contextlib import contextmanager

from fastapi import HTTPException

from fotoviewer.services.image_files import ImageNotFoundError


@contextmanager
def not_found_as_http_404(detail: str):
    try:
        yield
    except ImageNotFoundError:
        raise HTTPException(status_code=404, detail=detail)
