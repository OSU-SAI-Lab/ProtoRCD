"""FastAPI service exposing the validated triage pipeline to an external UI.

Design invariants
-----------------
* The frozen ``TriageSession`` (commit 08e0fa0) does all the science, unchanged. This
  server is orchestration + IO only.
* Feature stores and proposal metadata stay on the backend (OSC). Source images remain
  OSC-resident and their filesystem paths are never exposed to the client; the backend
  extracts and serves only the requested proposal crops. The client receives proposal
  ids, bboxes, and rendered crop JPEGs -- never feature vectors, image paths, or the
  image store itself.
* A session's whole lifecycle runs on a worker thread (see ``session_manager``); HTTP
  requests never block on human review. A worker crash puts the session in an explicit
  FAILED state (surfaced as HTTP 500) rather than leaving the UI waiting.
* Label submissions carry the pending batch's ``batch_id``; a stale/duplicate submission
  is rejected (HTTP 409) instead of being applied twice.

Endpoints
---------
    GET  /datasets                         -> configured datasets (names only)
    POST /sessions                         -> create+start a session (parks on calibration)
    GET  /sessions/{id}                    -> session state
    GET  /sessions/{id}/next-batch         -> the batch awaiting labels (calibration/review)
    POST /sessions/{id}/labels             -> submit labels; advances one phase
    GET  /sessions/{id}/triage             -> latest round's Resolve/Clear/Defer result
    GET  /crops/{id}/{proposal_id}         -> JPEG crop of a proposal
    DELETE /sessions/{id}                   -> discard a session

Dataset registry
----------------
Feature-store paths and image roots are configured server-side (a JSON file pointed to by
``PROTORCD_SERVING_CONFIG``, or the built-in empty registry). The UI only ever sends a
dataset *name*, so backend paths are never exposed.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from protorcd_pipeline.triage import (
    labels_from_coco,
    labels_from_coco_multi,
    load_pool,
    load_pool_multi,
)

from .crops import CropService
from .session_manager import SessionError, SessionManager, StaleBatchError

logger = logging.getLogger("protorcd.serving")

# Generic client-facing messages -- detailed causes (which may contain OSC paths) are
# logged server-side, never returned to the client.
_INTERNAL_ERROR = "internal server error"
_CROP_UNAVAILABLE = "crop unavailable"
_SESSION_FAILED = "session failed"

__all__ = ["app", "create_app", "DatasetConfig", "load_registry"]


# --------------------------------------------------------------------------- #
# Server-side dataset registry (paths never leave the backend)                 #
# --------------------------------------------------------------------------- #
class DatasetConfig(BaseModel):
    """Backend-only configuration for one servable dataset."""

    name: str
    bg_features: List[str]                       # one dir, or several for multi-split
    fg_features: List[str]
    coco: List[str] = Field(default_factory=list)  # optional; only for offline GT labels
    split_tags: Optional[List[str]] = None
    # Image root for crops. Single-root datasets set ``image_root``. Pooled multi-split
    # datasets (e.g. LoTE) whose per-split file_names are relative to different trees set
    # ``image_roots`` keyed by split tag (matching ``split_tags``); ``image_root`` then
    # acts as an optional default. Exactly one of the two must be present.
    image_root: Optional[str] = None
    image_roots: Optional[Dict[str, str]] = None
    iou_threshold: float = 0.25
    # Optional frozen density (kept-space .npy) to pin LoTE-scale tie-breaking exactly.
    density_cache: Optional[str] = None
    # Taxonomy: allowed foreground class id -> name (id 0 == background is always allowed
    # and need not be listed). Given inline here, OR loaded from ``labels_file`` (a JSON
    # object {id: name}). If neither is set the taxonomy is derived from the COCO
    # categories when ``coco`` is configured; if that too is absent, label validation is
    # skipped (any non-negative int is accepted) and a warning is logged.
    labels: Optional[Dict[int, str]] = None
    labels_file: Optional[str] = None


def load_registry(path: Optional[str] = None) -> Dict[str, DatasetConfig]:
    path = path or os.environ.get("PROTORCD_SERVING_CONFIG")
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return {d["name"]: DatasetConfig(**d) for d in data.get("datasets", [])}


def _resolve_taxonomy(
    cfg: DatasetConfig, coco_names: Optional[Dict[int, str]]
) -> Optional[Dict[int, str]]:
    """Determine the allowed foreground taxonomy for a dataset (id -> name), or None.

    Precedence: inline ``labels`` > ``labels_file`` > COCO category names. Background
    (id 0) is always allowed and is excluded from the returned foreground map. Returning
    None means "no taxonomy configured" -> label-id validation is skipped for this dataset.
    """
    raw: Optional[Dict] = None
    if cfg.labels:
        raw = dict(cfg.labels)
    elif cfg.labels_file:
        raw = json.loads(Path(cfg.labels_file).read_text())
    elif coco_names:
        raw = dict(coco_names)
    if raw is None:
        return None
    taxonomy = {int(k): str(v) for k, v in raw.items() if int(k) != 0}
    return taxonomy


# --------------------------------------------------------------------------- #
# Request / response models                                                    #
# --------------------------------------------------------------------------- #
class CreateSessionRequest(BaseModel):
    dataset: str
    seed: int = 3
    batch_size: int = 50
    calibration_size: int = 100
    max_rounds: Optional[int] = 16
    # If true, calibration labels come from the offline COCO GT (reproducibility mode)
    # instead of the reviewer; the calibration batch is then auto-answered server-side.
    offline_calibration: bool = False


class LabelsRequest(BaseModel):
    labels: Dict[str, int]     # proposal_id -> class (0 == background)
    batch_id: str              # the pending batch's id; required, validated as current


def _proposal_payload(proposals: List[dict]) -> List[dict]:
    return [
        {
            "proposal_id": str(p["proposal_id"]),
            "image_id": p.get("image_id"),
            "bbox_xywh": p.get("bbox_xywh"),
        }
        for p in proposals
    ]


def _batch_payload(batch) -> Optional[dict]:
    """Serialize a PendingBatch (or None) for the wire, including its batch_id."""
    if batch is None:
        return None
    return {
        "batch_id": batch.batch_id,
        "phase": batch.phase,
        "round": batch.round,
        "proposals": _proposal_payload(batch.proposals),
    }


# --------------------------------------------------------------------------- #
# App factory                                                                  #
# --------------------------------------------------------------------------- #
def create_app(registry: Optional[Dict[str, DatasetConfig]] = None) -> FastAPI:
    app = FastAPI(title="protorcd triage service", version="1.0")
    registry = registry if registry is not None else load_registry()
    manager = SessionManager()
    crop_services: Dict[str, CropService] = {}     # session_id -> CropService
    gt_labels_by_session: Dict[str, np.ndarray] = {}
    taxonomy_by_session: Dict[str, Optional[Dict[int, str]]] = {}

    def _build_pool_and_labels(cfg: DatasetConfig):
        multi = len(cfg.bg_features) > 1 or len(cfg.fg_features) > 1
        coco_names: Optional[Dict[int, str]] = None
        if multi:
            pool = load_pool_multi(cfg.bg_features, cfg.fg_features, split_tags=cfg.split_tags)
            if cfg.coco:
                gt, coco_names = labels_from_coco_multi(pool.metadata, cfg.coco, cfg.iou_threshold)
            else:
                gt = None
        else:
            pool = load_pool(cfg.bg_features[0], cfg.fg_features[0])
            if cfg.coco:
                gt, coco_names = labels_from_coco(pool.metadata, cfg.coco[0], cfg.iou_threshold)
            else:
                gt = None
        return pool, gt, coco_names

    def _dataset_taxonomy(cfg: DatasetConfig) -> Optional[Dict[int, str]]:
        """Taxonomy without building the pool -- for GET /datasets/{name}/labels.

        Uses inline labels / labels_file directly; falls back to COCO categories, reading
        just the categories block (not the whole annotation set).
        """
        if cfg.labels or cfg.labels_file:
            return _resolve_taxonomy(cfg, None)
        if cfg.coco:
            names: Dict[int, str] = {}
            for path in cfg.coco:
                data = json.loads(Path(path).read_text())
                names.update({int(c["id"]): str(c.get("name", c["id"]))
                              for c in data.get("categories", [])})
            return _resolve_taxonomy(cfg, names)
        return None

    def _validate_labels(taxonomy: Optional[Dict[int, str]], labels: Dict) -> None:
        """Reject labels that are negative or not in the dataset taxonomy (0 is allowed).

        If no taxonomy is configured, only the sign is checked (defensive minimum)."""
        allowed = set(taxonomy.keys()) if taxonomy is not None else None
        bad = []
        for pid, cls in labels.items():
            c = int(cls)
            if c < 0 or (c != 0 and allowed is not None and c not in allowed):
                bad.append((str(pid), c))
        if bad:
            preview = ", ".join(f"{p}={c}" for p, c in bad[:8])
            raise HTTPException(
                400,
                f"invalid label id(s) for this dataset: {preview}"
                f"{'...' if len(bad) > 8 else ''} (0=background; positives must be a "
                f"configured class id)",
            )

    @app.get("/datasets")
    def list_datasets():
        return {"datasets": sorted(registry.keys())}

    @app.get("/datasets/{dataset}/labels")
    def dataset_labels(dataset: str):
        if dataset not in registry:
            raise HTTPException(404, f"unknown dataset: {dataset}")
        taxonomy = _dataset_taxonomy(registry[dataset])
        return {
            "dataset": dataset,
            "background_id": 0,
            "validated": taxonomy is not None,
            "labels": None if taxonomy is None else {str(k): v for k, v in sorted(taxonomy.items())},
        }

    @app.post("/sessions")
    def create_session(req: CreateSessionRequest):
        if req.dataset not in registry:
            raise HTTPException(404, f"unknown dataset: {req.dataset}")
        cfg = registry[req.dataset]
        pool, gt, coco_names = _build_pool_and_labels(cfg)
        if gt is not None:
            pool.gt_labels = gt

        taxonomy = _resolve_taxonomy(cfg, coco_names)
        if taxonomy is None:
            logger.warning(
                "dataset %s has no taxonomy (labels/labels_file/coco); label-id "
                "validation is limited to sign checks for its sessions", cfg.name
            )

        density = np.load(cfg.density_cache) if cfg.density_cache else None
        managed = manager.create(
            pool, seed=req.seed, batch_size=req.batch_size,
            calibration_size=req.calibration_size, image_root=cfg.image_root,
            precomputed_density=density, max_rounds=req.max_rounds,
        )
        crop_services[managed.session_id] = CropService(
            pool, cfg.image_root, image_roots=cfg.image_roots
        )
        taxonomy_by_session[managed.session_id] = taxonomy
        if gt is not None:
            gt_labels_by_session[managed.session_id] = gt

        # Wait for the first batch (calibration) to be acquired.
        try:
            batch = managed.wait_for_batch()
            if req.offline_calibration and gt is not None and batch is not None:
                # Auto-answer calibration from GT so a UI can start at review round 1.
                idx = pool.id_to_index
                cal_labels = {str(pid): int(gt[idx[pid]]) for pid in batch.proposal_ids}
                managed.submit_labels(cal_labels, batch_id=batch.batch_id)
                batch = managed.wait_for_batch()
        except SessionError as exc:
            logger.exception("session %s failed during startup: %s", managed.session_id, exc)
            raise HTTPException(500, _SESSION_FAILED)

        return {
            "session_id": managed.session_id,
            "state": managed.state(),
            "labels": None if taxonomy is None else {str(k): v for k, v in sorted(taxonomy.items())},
            "next_batch": _batch_payload(batch),
        }

    def _managed(session_id: str):
        try:
            return manager.get(session_id)
        except KeyError:
            raise HTTPException(404, f"unknown session: {session_id}")

    @app.get("/sessions/{session_id}")
    def get_state(session_id: str):
        return _managed(session_id).state()

    @app.get("/sessions/{session_id}/next-batch")
    def next_batch(session_id: str):
        managed = _managed(session_id)
        try:
            batch = managed.wait_for_batch()
        except SessionError as exc:
            logger.exception("session %s failed: %s", session_id, exc)
            raise HTTPException(500, _SESSION_FAILED)
        except TimeoutError:
            raise HTTPException(504, "batch not ready yet; retry")
        if batch is None:
            return {"finished": True, "proposals": []}
        payload = _batch_payload(batch)
        payload["finished"] = False
        return payload

    @app.post("/sessions/{session_id}/labels")
    def submit_labels(session_id: str, req: LabelsRequest):
        managed = _managed(session_id)
        # Reject out-of-taxonomy / negative label ids BEFORE they can enter the prototype
        # bank (guards against a UI bug submitting an arbitrary integer class). Runs before
        # batch consumption, so a rejected submission leaves the batch answerable.
        _validate_labels(taxonomy_by_session.get(session_id), req.labels)
        # Map string ids from JSON back to the pool's id type.
        index = managed.pool.id_to_index
        remap: Dict = {}
        for k, v in req.labels.items():
            pid = k if k in index else _coerce_id(k, index)
            remap[pid] = int(v)
        try:
            info = managed.submit_labels(remap, batch_id=req.batch_id)
        except StaleBatchError as exc:
            raise HTTPException(409, str(exc))      # conflict: stale/duplicate submission
        except KeyError as exc:
            raise HTTPException(400, str(exc))      # incomplete labels (no OSC paths here)
        except SessionError as exc:
            logger.exception("session %s failed on submit: %s", session_id, exc)
            raise HTTPException(500, _SESSION_FAILED)
        except TimeoutError:
            # The labels were already accepted and the batch consumed; a resubmit would
            # 409. The client must NOT retry -- only poll session status.
            raise HTTPException(
                504,
                "labels accepted but the round is still processing; "
                "do not resubmit -- poll GET /sessions/{id} until status leaves 'advancing'",
            )
        result = None
        if info is not None and info.triage is not None:
            result = _triage_payload(info)
        # Fetch the next batch (or finished) so the UI has one round-trip.
        try:
            batch = managed.wait_for_batch()
        except SessionError as exc:
            logger.exception("session %s failed fetching next batch: %s", session_id, exc)
            raise HTTPException(500, _SESSION_FAILED)
        except TimeoutError:
            raise HTTPException(504, "batch not ready yet; retry")
        return {
            "state": managed.state(),
            "triage": result,
            "next_batch": _batch_payload(batch),
        }

    @app.get("/sessions/{session_id}/triage")
    def latest_triage(session_id: str):
        managed = _managed(session_id)
        info = managed._last_round
        if info is None or info.triage is None:
            return {"round": managed.session.round, "triage": None}
        return _triage_payload(info)

    @app.get("/crops/{session_id}/{proposal_id}")
    def get_crop(session_id: str, proposal_id: str):
        if session_id not in crop_services:
            raise HTTPException(404, f"unknown session: {session_id}")
        svc = crop_services[session_id]
        index = svc.pool.id_to_index
        pid = proposal_id if proposal_id in index else _coerce_id(proposal_id, index)
        try:
            data = svc.crop_bytes(pid)
        except KeyError:
            raise HTTPException(404, f"unknown proposal: {proposal_id}")
        except FileNotFoundError as exc:
            # Do NOT echo the resolved path (it reveals the OSC image root); log it only.
            logger.warning("crop source missing for %s/%s: %s", session_id, proposal_id, exc)
            raise HTTPException(404, _CROP_UNAVAILABLE)
        except Exception as exc:
            logger.exception("crop failed for %s/%s: %s", session_id, proposal_id, exc)
            raise HTTPException(500, _CROP_UNAVAILABLE)
        return Response(content=data, media_type="image/jpeg")

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        manager.remove(session_id)
        crop_services.pop(session_id, None)
        gt_labels_by_session.pop(session_id, None)
        taxonomy_by_session.pop(session_id, None)
        return {"deleted": session_id}

    return app


def _coerce_id(key: str, index: Dict):
    """Best-effort map a stringified proposal id back to the pool's id type."""
    if key in index:
        return key
    try:
        as_int = int(key)
        if as_int in index:
            return as_int
    except (TypeError, ValueError):
        pass
    return key  # let the caller raise a clean 404/400 if still unknown


def _triage_payload(info) -> dict:
    res = info.triage
    return {
        "round": info.round,
        "retired": info.retired,
        "counts": res.counts(),
        "decisions": [
            {
                "proposal_id": str(pid),
                "decision": dec,
                "predicted_class": int(pc),
            }
            for pid, dec, pc in zip(res.proposal_ids, res.decisions, res.predicted_class.tolist())
        ],
    }


# Default app (reads registry from PROTORCD_SERVING_CONFIG if set).
app = create_app()
