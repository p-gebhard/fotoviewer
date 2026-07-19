import os
from pathlib import Path

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


class ImageNotFoundError(Exception):
    pass


def _resolve_within_base(images_path: str, relative_path: str) -> Path:
    base = Path(images_path).resolve()
    full_path = (base / relative_path).resolve()
    if not full_path.is_relative_to(base):
        raise ImageNotFoundError(relative_path)
    return full_path


def resolve_image_path(images_path: str, relative_path: str) -> Path:
    full_path = _resolve_within_base(images_path, relative_path)
    if not full_path.is_file():
        raise ImageNotFoundError(relative_path)
    return full_path


def resolve_directory_path(images_path: str, relative_path: str) -> Path:
    full_path = _resolve_within_base(images_path, relative_path)
    if not full_path.is_dir():
        raise ImageNotFoundError(relative_path)
    return full_path


def find_image_paths(root: str) -> list[str]:
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if filename.startswith("."):
                continue
            if os.path.splitext(filename.lower())[-1] not in IMAGE_EXTENSIONS:
                continue
            image_paths.append(os.path.join(dirpath, filename))
    return image_paths
