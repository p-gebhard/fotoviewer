import json
import os
import threading

from PIL import Image

from fotoviewer.services.image_files import ImageNotFoundError, resolve_image_path
from fotoviewer.services.image_metadata import date_taken_from_exif

DATE_CACHE_FILENAME = "dates.json"


class DateCache:
    """Persistent cache of the EXIF capture date per photo, keyed by relative path and
    validated against the file's mtime and size, so modified photos are re-read
    automatically. Without a cache_file it only caches in memory."""

    def __init__(self, cache_file: str | None = None) -> None:
        self._cache_file = cache_file
        self._entries: dict[str, list] = {}
        self._dirty = False
        self._lock = threading.Lock()
        if cache_file:
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._entries = data
            except (OSError, ValueError):
                pass

    def get(self, images_path: str, relative_path: str) -> str | None:
        """Returns the EXIF date string ("YYYY:MM:DD HH:MM:SS") or None if the photo has
        no (readable) date. Raises ImageNotFoundError if the file does not exist."""
        full_path = resolve_image_path(images_path, relative_path)
        stat = full_path.stat()
        stamp = [stat.st_mtime_ns, stat.st_size]

        cached = self._entries.get(relative_path)
        if cached is not None and cached[:2] == stamp:
            return cached[2]

        try:
            with Image.open(full_path) as image:
                date_taken = date_taken_from_exif(image)
        except Exception:
            date_taken = None

        with self._lock:
            self._entries[relative_path] = [*stamp, date_taken]
            self._dirty = True
        return date_taken

    def warm(self, images_path: str, relative_paths) -> None:
        for relative_path in relative_paths:
            try:
                self.get(images_path, relative_path)
            except ImageNotFoundError:
                continue

    def prune(self, keep: set[str]) -> None:
        with self._lock:
            stale = [path for path in self._entries if path not in keep]
            for path in stale:
                del self._entries[path]
            if stale:
                self._dirty = True

    def save(self) -> None:
        if not self._cache_file:
            return
        with self._lock:
            if not self._dirty:
                return
            os.makedirs(os.path.dirname(self._cache_file) or ".", exist_ok=True)
            tmp_path = self._cache_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(self._entries, f)
            os.replace(tmp_path, self._cache_file)
            self._dirty = False
