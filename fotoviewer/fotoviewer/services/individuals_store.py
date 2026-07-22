import json
import uuid
from pathlib import Path

import numpy as np

from fotoviewer.services.identity import INDIVIDUALS_COLLECTION
from fotoviewer.services.indexing.index import Index


class IndividualEntryNotFoundError(Exception):
    pass


def _meta_path(index_dir: str) -> Path:
    return Path(index_dir) / "individuals_meta.json"


def _index_path(index_dir: str) -> Path:
    return Path(index_dir) / "individuals.npz"


def _load_metadata(index_dir: str, count: int) -> list[dict]:
    path = _meta_path(index_dir)
    meta = json.loads(path.read_text()) if path.exists() else []
    missing = count - len(meta)
    if missing > 0:
        meta += [{"id": str(uuid.uuid4()), "path": None, "bbox": None} for _ in range(missing)]
    return meta[:count]


def _save_metadata(index_dir: str, meta: list[dict]) -> None:
    _meta_path(index_dir).write_text(json.dumps(meta))


def _save_index(index_dir: str, individuals_index: Index) -> None:
    individuals_index.save(INDIVIDUALS_COLLECTION, str(_index_path(index_dir)))


def _rewrite(index_dir: str, individuals_index: Index, names: list[str], features: np.ndarray, meta: list[dict]) -> None:
    individuals_index.drop(INDIVIDUALS_COLLECTION)
    individuals_index.create(INDIVIDUALS_COLLECTION)
    if names:
        individuals_index.add_batch(INDIVIDUALS_COLLECTION, names, features)
    _save_index(index_dir, individuals_index)
    _save_metadata(index_dir, meta)


def append_entry(index_dir: str, individuals_index: Index, path: str, bbox: list[float]) -> dict:
    count = individuals_index.num_features(INDIVIDUALS_COLLECTION)
    meta = _load_metadata(index_dir, count - 1)
    entry = {"id": str(uuid.uuid4()), "path": path, "bbox": bbox}
    meta.append(entry)
    _save_metadata(index_dir, meta)
    return entry


def list_individuals(index_dir: str, individuals_index: Index) -> list[dict]:
    if not individuals_index.exists(INDIVIDUALS_COLLECTION):
        return []
    names = individuals_index.files(INDIVIDUALS_COLLECTION)
    meta = _load_metadata(index_dir, len(names))

    grouped: dict[str, list[dict]] = {}
    for name, entry in zip(names, meta):
        grouped.setdefault(str(name), []).append(entry)

    return [{"name": name, "entries": entries} for name, entries in grouped.items()]


def get_entry(index_dir: str, individuals_index: Index, entry_id: str) -> dict:
    names = individuals_index.files(INDIVIDUALS_COLLECTION) if individuals_index.exists(INDIVIDUALS_COLLECTION) else []
    meta = _load_metadata(index_dir, len(names))
    for entry in meta:
        if entry["id"] == entry_id:
            return entry
    raise IndividualEntryNotFoundError(entry_id)


def _position_of(meta: list[dict], entry_id: str) -> int:
    for i, entry in enumerate(meta):
        if entry["id"] == entry_id:
            return i
    raise IndividualEntryNotFoundError(entry_id)


def rename_entry(index_dir: str, individuals_index: Index, entry_id: str, new_name: str) -> None:
    names = list(individuals_index.files(INDIVIDUALS_COLLECTION))
    meta = _load_metadata(index_dir, len(names))
    position = _position_of(meta, entry_id)
    names[position] = new_name
    _rewrite(index_dir, individuals_index, names, individuals_index.features(INDIVIDUALS_COLLECTION), meta)


def delete_entry(index_dir: str, individuals_index: Index, entry_id: str) -> None:
    names = list(individuals_index.files(INDIVIDUALS_COLLECTION))
    meta = _load_metadata(index_dir, len(names))
    position = _position_of(meta, entry_id)

    del names[position]
    del meta[position]
    features = individuals_index.features(INDIVIDUALS_COLLECTION)
    remaining_features = np.delete(features, position, axis=0)
    _rewrite(index_dir, individuals_index, names, remaining_features, meta)


def rename_group(index_dir: str, individuals_index: Index, old_name: str, new_name: str) -> None:
    names = list(individuals_index.files(INDIVIDUALS_COLLECTION))
    meta = _load_metadata(index_dir, len(names))
    updated_names = [new_name if n == old_name else n for n in names]
    _rewrite(index_dir, individuals_index, updated_names, individuals_index.features(INDIVIDUALS_COLLECTION), meta)


def delete_group(index_dir: str, individuals_index: Index, name: str) -> None:
    names = list(individuals_index.files(INDIVIDUALS_COLLECTION))
    meta = _load_metadata(index_dir, len(names))
    keep = [i for i, n in enumerate(names) if n != name]

    kept_names = [names[i] for i in keep]
    kept_meta = [meta[i] for i in keep]
    features = individuals_index.features(INDIVIDUALS_COLLECTION)
    kept_features = features[keep] if keep else np.array([])
    _rewrite(index_dir, individuals_index, kept_names, kept_features, kept_meta)
