import os

import numpy as np

from fotoviewer.services.indexing.index import Index
from fotoviewer.services.similarity import cosine_scores

CHUNK_SIZE = 500


def _file_size(images_path: str, relative_path: str) -> int:
    try:
        return os.path.getsize(os.path.join(images_path, relative_path))
    except OSError:
        return 0


def find_duplicate_groups(image_index: Index, threshold: float, images_path: str) -> list[list[dict]]:
    if not image_index.exists("image") or image_index.num_features("image") == 0:
        return []

    features = image_index.features("image")
    files = image_index.files("image")
    n = len(files)

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    best_score = {}

    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk = features[start:end]
        remaining = features[start:]
        scores = cosine_scores(chunk, remaining)
        for local_i, global_i in enumerate(range(start, end)):
            row = scores[local_i]
            for local_j in np.where(row >= threshold)[0]:
                global_j = start + int(local_j)
                if global_j <= global_i:
                    continue
                union(global_i, global_j)
                score = float(row[local_j])
                best_score[global_i] = max(best_score.get(global_i, 0.0), score)
                best_score[global_j] = max(best_score.get(global_j, 0.0), score)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    result = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        entries = [
            {"file": str(files[i]), "score": best_score.get(i, 0.0), "size_bytes": _file_size(images_path, str(files[i]))}
            for i in indices
        ]
        largest_size = max(entry["size_bytes"] for entry in entries)
        for entry in entries:
            entry["suggest_delete"] = entry["size_bytes"] < largest_size
        group = sorted(entries, key=lambda entry: entry["score"], reverse=True)
        result.append(group)

    result.sort(key=lambda group: max(entry["score"] for entry in group), reverse=True)
    return result
