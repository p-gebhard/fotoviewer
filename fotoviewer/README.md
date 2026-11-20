# Foto Viewer

Local semantic photo search with object detection. All models (SigLIP2 for text/image search, RT-DETR for object detection) run entirely locally — there is no cloud API. Model weights are downloaded once from Hugging Face on first start and cached locally afterwards.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The Python version is pinned to 3.14 via `.python-version` (there are no `torch` wheels for newer versions yet); uv downloads it automatically if needed.

```
git clone <repository-url>
cd fotoviewer/fotoviewer
uv sync
cp .env.example .env   # set FOTOVIEWER_IMAGES_PATH to your photo folder
```

`.env` is loaded automatically on startup (see [fotoviewer/core/config.py](fotoviewer/core/config.py)) and is gitignored — `.env.example` serves as a template listing all available settings. Relative paths in the settings are resolved against the project directory (the one containing `pyproject.toml` and `.env`), not the current working directory.

## First run

1. **Build the search index** — this runs every photo through the embedding model once, which can take a while for large libraries (subsequent runs only process new or modified photos):

   ```
   uv run fotoviewer-create-index
   ```

   Optionally also build the object index used by `/filter` (see [Library filter](#library-filter-filter)):

   ```
   uv run fotoviewer-index-detections
   ```

2. **Start the app:**

   ```
   uv run uvicorn fotoviewer.main:app --reload
   ```

   The app is then available at `http://127.0.0.1:8000`.

On the very first start, the model weights (several GB) are downloaded from Hugging Face. The app also starts without an index; the search page then shows a hint, and the index can be built from the app via `/sources` ("Reindex now") instead of the command line.

## Running

```
uv run uvicorn fotoviewer.main:app --reload
```

By default the app only binds to `127.0.0.1` and has **no authentication** — a deliberate choice for a local single-user tool. Before binding to `0.0.0.0` or exposing it on a network, put an auth layer in front of it (e.g. a reverse proxy with basic auth).

## Pages

| Route | Content |
|---|---|
| `/` | Semantic text/image search |
| `/image?path=...` | Detail page: metadata, object detection, tagging individuals |
| `/folders` | File browser for the configured photo library |
| `/sources` | Overview of indices, models and thresholds; manual reindexing of the library |
| `/individuals` | Manage the gallery of individuals (rename, delete) |
| `/duplicates` | Review possible duplicate groups and delete selected files (see below) |
| `/filter` | Filter the library by folder, individual and date range (see below) |
| `/import` | Copy photos from any source folder into the library and reindex automatically |

## Models

All models run locally via [`transformers`](https://github.com/huggingface/transformers); the project uses no other models.

| Model | Hugging Face ID (default) | Purpose | Code |
|---|---|---|---|
| SigLIP2 | `google/siglip2-base-patch16-224` | Text/image embeddings for semantic search **and** the basis for recognizing individuals (embedding of the cropped detection) | [fotoviewer/services/image_processor.py](fotoviewer/services/image_processor.py) |
| RT-DETR | `PekingU/rtdetr_r50vd` | Object detection (91 COCO classes) — both for detection on a stored photo's detail page and for the WebSocket upload flow | [fotoviewer/services/object_detector/object_detector.py](fotoviewer/services/object_detector/object_detector.py) |

Both model names can be swapped via `FOTOVIEWER_EMBEDDING_MODEL_NAME` / `FOTOVIEWER_DETECTION_MODEL_NAME` (see configuration below). Recognizing individuals does not use a dedicated model, only SigLIP2 on cropped detections (see [fotoviewer/services/identity.py](fotoviewer/services/identity.py) and the limitations below).

## Configuration

Via environment variables prefixed with `FOTOVIEWER_` (see [fotoviewer/core/config.py](fotoviewer/core/config.py)):

| Variable | Default | Meaning |
|---|---|---|
| `FOTOVIEWER_IMAGES_PATH` | – (required) | Path to the photo library |
| `FOTOVIEWER_INDEX_DIR` | `../indices` | Where the embedding indices are stored |
| `FOTOVIEWER_EMBEDDING_MODEL_NAME` | `google/siglip2-base-patch16-224` | Model for text/image search |
| `FOTOVIEWER_DETECTION_MODEL_NAME` | `PekingU/rtdetr_r50vd` | Model for object detection |
| `FOTOVIEWER_IDENTITY_MATCH_THRESHOLD` | `0.75` | Threshold for recognizing individuals (see limitations below) |
| `FOTOVIEWER_DUPLICATE_MATCH_THRESHOLD` | `0.95` | Threshold for duplicate detection on `/duplicates` |
| `FOTOVIEWER_DETECTION_THRESHOLD` | `0.7` | Minimum score for RT-DETR to report a detection |
| `FOTOVIEWER_DEVICE` | – (auto) | Force `cuda` or `cpu`; leave empty for automatic detection |
| `FOTOVIEWER_LANGUAGE` | `en` | UI language: `en` or `de` (see below) |

Additionally `HF_HOME` to put the Hugging Face model cache on a drive with enough space (default is `~/.cache/huggingface`; the models add up to several GB).

## Language

The UI language is set via `FOTOVIEWER_LANGUAGE` (`en` or `de`) and applies to all pages, client-side messages, error messages shown in the UI and the `manifest.txt` written on duplicate cleanup. Texts live in [fotoviewer/locales/](fotoviewer/locales/) as one JSON file per language ([fotoviewer/core/i18n.py](fotoviewer/core/i18n.py)); missing keys fall back to English. Keys prefixed with `js.` are also made available to client-side JavaScript.

To add a language, copy `en.json` to `<code>.json`, translate the values and add the code to `SUPPORTED_LANGUAGES` in `i18n.py` and to the `language` field in [fotoviewer/core/config.py](fotoviewer/core/config.py). The tests check that all languages define the same keys and placeholders and that every key used in a template exists.

## Building the search index

```
uv run fotoviewer-create-index
uv run fotoviewer-search <image-path>
```

Indexing is **incremental**: next to each index (`image.npz`, `detections.npz`) a manifest (`*.npz.manifest.json`) stores the modification time and size of every processed photo. Subsequent runs only send new and modified photos through the models, entries of deleted photos are dropped, and everything else is kept. Unreadable photos are not recorded in the manifest and are retried on the next run.

A full rebuild happens automatically when no manifest exists yet (i.e. on the first run after upgrading) or when the model or detection threshold has changed. To force one manually, use `--full`:

```
uv run fotoviewer-create-index --full
uv run fotoviewer-index-detections --full
```

Alternatively, the library can be updated from within the app on `/sources` ("Reindex now"), optionally with "Full rebuild". The import on `/import` uses the same incremental process and therefore only processes the newly copied photos.

## Deleting duplicates (`/duplicates`)

Deleted duplicates are not removed permanently but moved to `.duplicates/<timestamp>/` inside the photo library, under their original relative path. Each run also writes a `manifest.txt` listing all moved files. To restore a file, simply move it back into the library under the same relative path. Files are only deleted for good once the `.duplicates` folder is emptied manually.

## Library filter (`/filter`)

To make the whole library filterable by folder, individual and year taken (rather than only finding individual photos via text/image search), there is an additional, optional indexing step:

```
uv run fotoviewer-index-detections
```

This runs RT-DETR once over every photo in the library and stores only the **embeddings of the detected object crops** (no names) in `detections.npz` — the actual identity matching happens live on every filter request against the current gallery of individuals ([fotoviewer/services/identity.py](fotoviewer/services/identity.py), [fotoviewer/services/library_filter.py](fotoviewer/services/library_filter.py)). As a result, a newly tagged individual takes effect in the filter immediately without rerunning `fotoviewer-index-detections` — the command only needs to run again when new photos are added to the library (thanks to incremental indexing, only the new photos are processed then).

The date range filter reads the capture date (EXIF `DateTimeOriginal`, falling back to `DateTime`) from the `dates.json` cache in the index directory. The cache is prefilled during indexing; missing or stale entries (changed modification time/size) are read on demand while filtering ([fotoviewer/services/date_cache.py](fotoviewer/services/date_cache.py)). Photos without a (reliable) EXIF date are excluded when a date range is active rather than guessed.

## Thumbnails

The grid views (search, folders, filter, duplicates) load downscaled thumbnails via `/thumbnails?path=...` instead of the originals. They are generated on first access (JPEG, max. 400 px, EXIF orientation applied) and cached under `thumbnails/` in the index directory ([fotoviewer/services/thumbnails.py](fotoviewer/services/thumbnails.py)). When a photo changes, a new thumbnail is created automatically; old thumbnails are left behind, and the folder can be safely deleted at any time. The detail page still shows the original.

## Tests

```
uv run pytest
```

## Known limitations

### Recognizing individuals is unreliable

On the image detail page, a detected box can be given a name (e.g. "Rex" instead of just "dog"). Future detections are then matched via cosine similarity against the resulting gallery of reference embeddings ([fotoviewer/services/identity.py](fotoviewer/services/identity.py)).

**This works mechanically, but is not reliable enough for real use in practice.** In a test with one reference image ("Rex") against a completely different photo with other people and animals, *all* eleven objects detected on it matched "Rex" with scores between 0.79 and 0.85 — only slightly below the score of the actual self-match (0.9999) and well above the default threshold of 0.75.

**Cause:** SigLIP2 (`FOTOVIEWER_EMBEDDING_MODEL_NAME`) is a general-purpose vision-language embedding trained on coarse semantic similarity ("this is a person", "this is a horse"), not on fine-grained identity discrimination between two different people or animals. Generic bounding-box crops end up closer together in embedding space than reliable recognition would require — pose, lighting and background affect the embedding more than identity itself.

**Implemented mitigation:** Instead of comparing only the single most similar reference crop, `identify_crop` ([fotoviewer/services/identity.py](fotoviewer/services/identity.py)) averages the score over *all* reference photos of a name and then compares the averages of the candidate individuals. A single, coincidentally very similar false match is thus diluted by the other, less similar reference photos of the same name — with only one reference, behavior is unchanged. This reduces the problem described above but does not solve it fundamentally: with consistently misclassified reference photos (e.g. when all references of a wrong name happen to look similar), the false recognition remains.

**Further possible steps** (not implemented yet):
- Raise the threshold significantly (e.g. 0.95+) — at the risk of false negatives for genuine matches in a different pose/lighting.
- Require a minimum number of reference photos per individual before a name is considered as a match at all.
- Add dedicated embeddings for people (face recognition) — SigLIP2 is not the right tool for this, but it would mean an additional model dependency.

### Object detection is limited to 91 COCO classes

RT-DETR only recognizes the fixed COCO categories (e.g. `person`, `dog`, `horse`). Objects outside this list (e.g. camels) are assigned to the closest known class instead of being named correctly.
