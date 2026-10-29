"""HTTP service layer exposing the validated triage pipeline to an external UI.

Orchestration and IO only -- the frozen ``protorcd_pipeline.triage`` algorithms (commit
08e0fa0) are used verbatim. Feature stores and proposal metadata stay on the backend;
source images remain OSC-resident and their filesystem paths are never exposed to the
client. The client sees only proposal ids, bboxes, and rendered crop JPEGs.

    from serving.server import create_app
    app = create_app()   # a FastAPI app; run with `uvicorn serving.server:app`
"""
from .crops import CropService
from .session_manager import ManagedSession, PendingBatch, QueueLabelProvider, SessionManager

__all__ = [
    "CropService",
    "SessionManager",
    "ManagedSession",
    "PendingBatch",
    "QueueLabelProvider",
]
