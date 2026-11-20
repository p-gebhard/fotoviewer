import shutil
from datetime import datetime
from pathlib import Path

from fotoviewer.core.i18n import Translator
from fotoviewer.services.image_files import ImageNotFoundError, resolve_image_path


def _write_manifest(trash_root: Path, moved_paths: list[str], timestamp: datetime, t: Translator) -> None:
    lines = [
        t("duplicates.manifest_title", timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S")),
        t("duplicates.manifest_origin"),
        t("duplicates.manifest_restore"),
        "",
        *moved_paths,
    ]
    (trash_root / "manifest.txt").write_text("\n".join(lines) + "\n")


def move_to_trash(images_path: str, relative_paths: list[str], translator: Translator | None = None) -> dict:
    timestamp = datetime.now()
    trash_root = Path(images_path) / ".duplicates" / timestamp.strftime("%Y%m%d-%H%M%S")

    moved = []
    failed = []
    for relative_path in relative_paths:
        try:
            full_path = resolve_image_path(images_path, relative_path)
        except ImageNotFoundError:
            failed.append(relative_path)
            continue

        destination = trash_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(full_path), str(destination))
            moved.append(relative_path)
        except OSError:
            failed.append(relative_path)

    if moved:
        _write_manifest(trash_root, moved, timestamp, translator or Translator("en"))

    return {"moved": moved, "failed": failed}
