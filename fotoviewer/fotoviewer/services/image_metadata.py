from pathlib import Path
from typing import Any

from PIL import ExifTags, Image


class ImageNotFoundError(Exception):
    pass


def get_image_metadata(images_path: str, relative_path: str) -> dict[str, Any]:
    base = Path(images_path).resolve()
    full_path = (base / relative_path).resolve()
    if not full_path.is_relative_to(base) or not full_path.is_file():
        raise ImageNotFoundError(relative_path)

    with Image.open(full_path) as image:
        width, height = image.size
        image_format = image.format
        exif = {
            ExifTags.TAGS.get(tag_id, str(tag_id)): value
            for tag_id, value in image.getexif().items()
        }

    return {
        "path": relative_path,
        "width": width,
        "height": height,
        "format": image_format,
        "file_size_bytes": full_path.stat().st_size,
        "date_taken": exif.get("DateTimeOriginal") or exif.get("DateTime"),
        "camera_make": exif.get("Make"),
        "camera_model": exif.get("Model"),
    }
