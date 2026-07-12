import numpy as np
from typing import Dict, List, Any, Optional


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

    def save(self, name: str, filename: str) -> None:
        index = self._loaded_indices[name]
        with open(filename, 'wb') as f:
            np.savez(f, features=np.array(index["features"], dtype=np.float32), files=np.array(index["files"], dtype=str))

    def num_features(self, name: str) -> int:
        return len(self._loaded_indices[name]["features"])

    def add(self, name: str, file: str, features: np.ndarray) -> None:
        index = self._loaded_indices[name]
        index["files"] = np.hstack([index["files"], file])
        if index["features"].size:
            index["features"] = np.vstack([index["features"], features])
        else:
            index["features"] = features

    def search(self, name: str, embedding: np.ndarray, k: int) -> List[Dict[str, Any]]:
        catalog_features = self._loaded_indices[name]["features"]
        files = self._loaded_indices[name]["files"]
        scores = (1 + np.dot(catalog_features, embedding.T).T[0]) / 2
        sorted_label_indices = np.argsort(scores)[::-1][:k]
        scores = scores.astype(dtype=float)
        return [
            {"file": files[idx], "score": scores[idx]}
            for idx in sorted_label_indices
        ]
