import shutil
from datetime import datetime
from pathlib import Path

from fotoviewer.services.image_files import ImageNotFoundError, resolve_image_path


def _write_manifest(trash_root: Path, moved_paths: list[str], timestamp: datetime) -> None:
    lines = [
        f"# Duplikat-Aufräumung vom {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        "# Diese Dateien lagen ursprünglich unter dem hier gezeigten relativen Pfad in der Fotobibliothek.",
        "# Zum Wiederherstellen: Datei aus diesem Ordner unter demselben relativen Pfad zurück in die Bibliothek verschieben.",
        "",
        *moved_paths,
    ]
    (trash_root / "manifest.txt").write_text("\n".join(lines) + "\n")


def move_to_trash(images_path: str, relative_paths: list[str]) -> dict:
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
        _write_manifest(trash_root, moved, timestamp)

    return {"moved": moved, "failed": failed}
