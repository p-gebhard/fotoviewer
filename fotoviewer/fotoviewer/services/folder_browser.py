import os

from fotoviewer.services.image_files import IMAGE_EXTENSIONS, find_image_paths, resolve_directory_path


def list_folder(images_path: str, relative_path: str) -> dict:
    relative_path = relative_path.strip("/")
    full_path = resolve_directory_path(images_path, relative_path)

    folders = []
    images = []
    for entry in sorted(os.listdir(full_path)):
        if entry.startswith("."):
            continue
        entry_path = full_path / entry
        if entry_path.is_dir():
            entry_relative_path = f"{relative_path}/{entry}" if relative_path else entry
            folders.append({
                "name": entry,
                "path": entry_relative_path,
                "image_count": len(find_image_paths(str(entry_path))),
            })
        elif os.path.splitext(entry.lower())[-1] in IMAGE_EXTENSIONS:
            images.append(entry)

    breadcrumbs = []
    if relative_path:
        accumulated = []
        for part in relative_path.split("/"):
            accumulated.append(part)
            breadcrumbs.append({"name": part, "path": "/".join(accumulated)})

    return {
        "path": relative_path,
        "breadcrumbs": breadcrumbs,
        "folders": folders,
        "images": images,
    }
