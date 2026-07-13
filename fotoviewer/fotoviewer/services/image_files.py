from pathlib import Path


class ImageNotFoundError(Exception):
    pass


def resolve_image_path(images_path: str, relative_path: str) -> Path:
    base = Path(images_path).resolve()
    full_path = (base / relative_path).resolve()
    if not full_path.is_relative_to(base) or not full_path.is_file():
        raise ImageNotFoundError(relative_path)
    return full_path
