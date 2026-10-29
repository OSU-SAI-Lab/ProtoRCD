"""Label providers -- how reviewed labels reach the triage loop (stage 3, part 4).

This is the seam that replaces "look the answer up from GT boxes." Acquisition selects
proposal ids; the loop asks a ``LabelProvider`` for their labels; triage consumes them.
Background is always class 0; any positive integer is a foreground species id.

Three providers ship here:

  CallbackLabelProvider  -- primary UI mode. You pass a function
                            ``request(list[proposal_id]) -> {proposal_id: label}``.
                            A Gradio/Streamlit/web UI implements this by showing the
                            crops and returning the reviewer's choices.

  FileLabelProvider      -- built ON TOP of the callback interface for out-of-process
                            UIs. Each round it writes the pending proposal ids to
                            ``<dir>/round_<r>_query.json`` and blocks until the UI
                            writes ``<dir>/round_<r>_labels.json`` back.

  GTLabelProvider        -- offline / reproducibility mode. Labels proposals by
                            IoU-matching them to COCO ground-truth boxes (the research
                            behavior), so paper numbers can be reproduced without a UI.

All providers expose the same method:
    get_labels(proposal_ids: Sequence[Hashable]) -> dict[proposal_id, int]
returning an integer label for every requested id (0 == background).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "LabelProvider",
    "CallbackLabelProvider",
    "FileLabelProvider",
    "GTLabelProvider",
]


class LabelProvider:
    """Interface: return an integer label (0 == background) for each proposal id."""

    def get_labels(self, proposal_ids: Sequence[Hashable]) -> Dict[Hashable, int]:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # optional cleanup hook
        pass


class CallbackLabelProvider(LabelProvider):
    """Primary UI mode: delegate to an in-process ``request`` callback.

    ``request`` receives the list of proposal ids to review and must return a mapping
    from each id to its integer label (0 == background). Missing ids raise -- the loop
    must not proceed with unlabeled acquisitions.
    """

    def __init__(self, request: Callable[[Sequence[Hashable]], Mapping[Hashable, int]]):
        self._request = request

    def get_labels(self, proposal_ids: Sequence[Hashable]) -> Dict[Hashable, int]:
        ids = list(proposal_ids)
        if not ids:
            return {}
        result = dict(self._request(ids))
        missing = [pid for pid in ids if pid not in result]
        if missing:
            raise KeyError(f"LabelProvider callback did not return labels for: {missing[:8]}"
                           f"{'...' if len(missing) > 8 else ''}")
        return {pid: int(result[pid]) for pid in ids}


class FileLabelProvider(LabelProvider):
    """Out-of-process UI mode: exchange JSON files per round.

    For each batch it writes ``round_<r>_query.json`` (a list of proposal ids to
    review) into ``exchange_dir`` and polls for ``round_<r>_labels.json``, a JSON object
    ``{proposal_id: label}`` the UI writes back. Blocks (polling every ``poll_seconds``)
    until every requested id has a label, or ``timeout_seconds`` elapses.
    """

    def __init__(
        self,
        exchange_dir: str | Path,
        *,
        poll_seconds: float = 1.0,
        timeout_seconds: Optional[float] = None,
        start_round: int = 1,
    ):
        self.dir = Path(exchange_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = float(poll_seconds)
        self.timeout_seconds = timeout_seconds
        self._round = int(start_round)

    def get_labels(self, proposal_ids: Sequence[Hashable]) -> Dict[Hashable, int]:
        ids = [str(pid) for pid in proposal_ids]
        if not ids:
            return {}
        r = self._round
        query_path = self.dir / f"round_{r}_query.json"
        labels_path = self.dir / f"round_{r}_labels.json"
        query_path.write_text(json.dumps({"round": r, "proposal_ids": ids}, indent=2))

        waited = 0.0
        while True:
            if labels_path.exists():
                try:
                    payload = json.loads(labels_path.read_text())
                except json.JSONDecodeError:
                    payload = None  # partial write; keep polling
                if isinstance(payload, dict):
                    raw = payload.get("labels", payload)
                    if all(pid in raw for pid in ids):
                        self._round += 1
                        # map back to the original id objects
                        out: Dict[Hashable, int] = {}
                        for orig, s in zip(proposal_ids, ids):
                            out[orig] = int(raw[s])
                        return out
            if self.timeout_seconds is not None and waited >= self.timeout_seconds:
                raise TimeoutError(
                    f"No labels for round {r} after {waited:.0f}s "
                    f"(expected {labels_path})"
                )
            time.sleep(self.poll_seconds)
            waited += self.poll_seconds


class GTLabelProvider(LabelProvider):
    """Offline reproducibility mode: label proposals by IoU-matching to GT boxes.

    This reproduces the research behavior (``label_metadata`` in
    ``evaluate_active_triage_features.py``): a proposal gets the category of the
    ground-truth box it overlaps most, if that IoU >= ``iou_threshold``; otherwise 0
    (background). Use it to reproduce paper numbers or to unit-test the loop on labeled
    data -- NOT in a live UI deployment.

    ``labels`` is a precomputed (N,) int array aligned to the proposal pool order; it is
    typically built by :func:`labels_from_coco`.
    """

    def __init__(self, labels: np.ndarray, id_to_index: Optional[Mapping[Hashable, int]] = None):
        self._labels = np.asarray(labels, dtype=np.int64)
        self._id_to_index = dict(id_to_index) if id_to_index is not None else None

    def get_labels(self, proposal_ids: Sequence[Hashable]) -> Dict[Hashable, int]:
        out: Dict[Hashable, int] = {}
        for pid in proposal_ids:
            if self._id_to_index is not None:
                idx = self._id_to_index[pid]
            else:
                idx = int(pid)  # ids ARE pool indices
            out[pid] = int(self._labels[idx])
        return out

    @property
    def labels(self) -> np.ndarray:
        return self._labels
