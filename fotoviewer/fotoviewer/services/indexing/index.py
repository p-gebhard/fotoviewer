import os
import numpy as np
from typing import Dict, List, Any, Optional

from fotoviewer.services.similarity import cosine_scores


class Index:

    def __init__(self):
        self._loaded_indices = {}

    def exists(self, name: str) -> bool:
        return name in self._loaded_indices

    def create(self, name: str) -> None:
        new_collection = {
            "files": np.array([]),
            "features": np.array([]),
        }
        self._loaded_indices[name] = new_collection

    def drop(self, name: str) -> None:
        del self._loaded_indices[name]

    def load(self, name: str, filename: str) -> None:
        container = np.load(file=filename)
        self._loaded_indices[name] = {
            "files": container["files"],
            "features": container["features"],
        }

    def load_or_create(self, name: str, filename: str) -> None:
        if os.path.exists(filename):
            self.load(name, filename)
        else:
            self.create(name)

    def save(self, name: str, filename: str) -> None:
        index = self._loaded_indices[name]
        with open(filename, 'wb') as f:
            np.savez(f, features=np.array(index["features"], dtype=np.float32), files=np.array(index["files"], dtype=str))

    def num_features(self, name: str) -> int:
        return len(self._loaded_indices[name]["features"])

    def files(self, name: str) -> np.ndarray:
        return self._loaded_indices[name]["files"]

    def features(self, name: str) -> np.ndarray:
        return self._loaded_indices[name]["features"]

    def add(self, name: str, file: str, features: np.ndarray) -> None:
        index = self._loaded_indices[name]
        index["files"] = np.hstack([index["files"], file])
        if index["features"].size:
            index["features"] = np.vstack([index["features"], features])
        else:
            index["features"] = features

    def add_batch(self, name: str, files: List[str], features: np.ndarray) -> None:
        index = self._loaded_indices[name]
        new_files = np.array(files, dtype=str)
        if index["features"].size:
            index["features"] = np.vstack([index["features"], features])
            index["files"] = np.hstack([index["files"], new_files])
        else:
            index["features"] = features
            index["files"] = new_files

    def remove_entries(self, name: str, values_to_remove: set) -> None:
        files = list(self._loaded_indices[name]["files"])
        keep = [i for i, f in enumerate(files) if str(f) not in values_to_remove]
        if not keep:
            self.create(name)
            return
        kept_files = [files[i] for i in keep]
        kept_features = self._loaded_indices[name]["features"][keep]
        self.drop(name)
        self.create(name)
        self.add_batch(name, kept_files, kept_features)

    def search(self, name: str, embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        catalog_features = self._loaded_indices[name]["features"]
        files = self._loaded_indices[name]["files"]
        scores = cosine_scores(catalog_features, embedding)[:, 0]
        sorted_label_indices = np.argsort(scores)[::-1][:k]
        scores = scores.astype(dtype=float)
        return [
            {"file": files[idx], "score": scores[idx]}
            for idx in sorted_label_indices
        ]
