import threading

from fastapi import APIRouter, Depends, HTTPException, Request

from fotoviewer.api.dependencies import get_settings
from fotoviewer.core.config import Settings
from fotoviewer.services.reindex import reload_indices, run_reindex

router = APIRouter()


def start_background_reindex(app, settings: Settings, full: bool = False) -> bool:
    """Starts an incremental (or full) reindex in a background thread. Returns False if an
    indexing run (reindex or import) is already active."""
    status = app.state.reindex_status
    if status["running"] or app.state.import_status["running"]:
        return False

    status.clear()
    status.update(running=True, phase="starting", error=None)

    def _run():
        run_reindex(
            settings.images_path,
            settings.index_dir,
            app.state.image_processor,
            app.state.object_detector,
            status,
            full=full,
            date_cache=app.state.date_cache,
        )
        reload_indices(app.state, settings.index_dir)

    threading.Thread(target=_run, daemon=True).start()
    return True


@router.post("/api/reindex", status_code=202)
def start_reindex(request: Request, full: bool = False, settings: Settings = Depends(get_settings)) -> dict:
    if not start_background_reindex(request.app, settings, full=full):
        raise HTTPException(status_code=409, detail=request.app.state.translator("errors.indexing_running"))
    return {"status": "started"}


@router.get("/api/reindex/status")
def reindex_status(request: Request) -> dict:
    return request.app.state.reindex_status
