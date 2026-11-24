"""Intermediate results of AI restoration steps.

An AI step (e.g. denoising) produces a new full-resolution image that the classic
sliders are then applied on top of, and further AI steps can be chained. These
intermediate images are kept as lossless PNG files outside the photo library until the
user saves a copy; they are removed automatically after a day.
"""

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

from fotoviewer.services.image_files import ImageNotFoundError

WORKING_DIRNAME = "edits"
MAX_AGE_SECONDS = 24 * 60 * 60

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


class WorkingImages:
    def __init__(self, directory: str) -> None:
        self._dir = Path(directory)
        self._lock = threading.Lock()

    def create(self, source_path: str, steps: list[str], image: Image.Image) -> dict:
        """Stores an intermediate image derived from the library photo source_path after
        the given AI steps and returns its description."""
        self._dir.mkdir(parents=True, exist_ok=True)
        working_id = uuid.uuid4().hex
        image_path = self._dir / f"{working_id}.png"
        tmp_path = self._dir / f".{working_id}.png.tmp"
        # compress_level=1: these files are short-lived, writing speed matters more than size.
        image.save(tmp_path, "PNG", compress_level=1)
        os.replace(tmp_path, image_path)

        meta = {"id": working_id, "source": source_path, "steps": steps, "width": image.width, "height": image.height}
        (self._dir / f"{working_id}.json").write_text(json.dumps(meta), encoding="utf-8")
        return meta

    def get(self, working_id: str, source_path: str) -> tuple[Path, dict]:
        """Returns (image path, meta) of an intermediate image. Raises ImageNotFoundError
        if it does not exist (anymore) or does not belong to source_path."""
        if not _ID_PATTERN.match(working_id):
            raise ImageNotFoundError(working_id)
        image_path = self._dir / f"{working_id}.png"
        meta_path = self._dir / f"{working_id}.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ImageNotFoundError(working_id)
        if meta.get("source") != source_path or not image_path.is_file():
            raise ImageNotFoundError(working_id)
        return image_path, meta

    def cleanup(self, max_age_seconds: float = MAX_AGE_SECONDS) -> int:
        """Deletes intermediate files older than max_age_seconds; returns the number removed."""
        if not self._dir.is_dir():
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        with self._lock:
            for path in self._dir.iterdir():
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError:
                    pass
        return removed
