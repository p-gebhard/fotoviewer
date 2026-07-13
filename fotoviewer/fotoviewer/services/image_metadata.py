from typing import Any

from PIL import ExifTags, Image

from fotoviewer.services.image_files import resolve_image_path


def get_image_metadata(images_path: str, relative_path: str) -> dict[str, Any]:
    full_path = resolve_image_path(images_path, relative_path)

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
