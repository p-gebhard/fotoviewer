import hashlib
import os
from pathlib import Path

from PIL import Image, ImageOps

from fotoviewer.services.image_files import resolve_image_path

THUMBNAIL_DIRNAME = "thumbnails"
THUMBNAIL_SIZE = 400


def get_thumbnail(images_path: str, cache_dir: str, relative_path: str, size: int = THUMBNAIL_SIZE) -> Path:
    """Returns the path of a cached JPEG thumbnail (longest side <= size), creating it on
    first access. The cache key includes mtime and size of the original, so a modified
    photo automatically gets a fresh thumbnail. Raises ImageNotFoundError."""
    full_path = resolve_image_path(images_path, relative_path)
    stat = full_path.stat()
    key = hashlib.sha1(f"{relative_path}|{stat.st_mtime_ns}|{stat.st_size}|{size}".encode()).hexdigest()

    thumb_path = Path(cache_dir) / key[:2] / f"{key}.jpg"
    if thumb_path.exists():
        return thumb_path

    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(full_path) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((size, size))
        if image.mode != "RGB":
            image = image.convert("RGB")
        tmp_path = thumb_path.with_suffix(f".{os.getpid()}.tmp")
        image.save(tmp_path, "JPEG", quality=85)
    os.replace(tmp_path, thumb_path)
    return thumb_path
