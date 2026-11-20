from typing import Any

from PIL import ExifTags, Image

from fotoviewer.services.image_files import resolve_image_path

_DATE_TIME_ORIGINAL = 0x9003
_DATE_TIME = 0x0132


def _exif_dict(image: Image.Image) -> dict[str, Any]:
    return {ExifTags.TAGS.get(tag_id, str(tag_id)): value for tag_id, value in image.getexif().items()}


def date_taken_from_exif(image: Image.Image) -> str | None:
    exif = image.getexif()
    # DateTimeOriginal lives in the Exif sub-IFD; some writers also put it in IFD0.
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    return exif_ifd.get(_DATE_TIME_ORIGINAL) or exif.get(_DATE_TIME_ORIGINAL) or exif.get(_DATE_TIME)


def get_image_metadata(images_path: str, relative_path: str) -> dict[str, Any]:
    full_path = resolve_image_path(images_path, relative_path)

    with Image.open(full_path) as image:
        width, height = image.size
        image_format = image.format
        exif = _exif_dict(image)
        date_taken = date_taken_from_exif(image)

    return {
        "path": relative_path,
        "width": width,
        "height": height,
        "format": image_format,
        "file_size_bytes": full_path.stat().st_size,
        "date_taken": date_taken,
        "camera_make": exif.get("Make"),
        "camera_model": exif.get("Model"),
    }
