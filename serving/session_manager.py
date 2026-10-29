"""Backend session management for serving the triage loop to an external UI.

This layer is PURE ORCHESTRATION around the frozen ``TriageSession`` (commit 08e0fa0).
It does not re-implement or modify the triage / acquisition / calibration algorithms: it
runs ``build_calibration_bank()`` and ``step()`` verbatim on a worker thread and bridges
human labels in over a thread-safe queue. The same code that passes the
research-equivalence regression drives the UI, so scientific parity is preserved by
construction.

Why a worker thread
-------------------
``build_calibration_bank()`` and ``step()`` are synchronous: they call
``provider.get_labels()`` inline and block until the reviewer answers. An HTTP request
cannot block for minutes waiting on a human. So each session runs its whole lifecycle on
one worker thread:

    review the calibration bank  ->  (park for labels)  ->  calibrate
    for each round:  acquire a batch  ->  (park for labels)  ->  triage + retire

Every time the loop needs labels it parks inside a :class:`QueueLabelProvider`; the HTTP
``submit_labels`` handler delivers them and the worker resumes. Nothing about the decision
math changes -- the thread is simply *where* the existing blocking calls wait.

The provider also records which loop phase each batch belongs to ("calibration" vs a
review round) purely so the UI can label the screen; it never affects the algorithm.
"""
from __future__ import annotations

import enum
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Sequence

import numpy as np

from protorcd_pipeline.triage import LabelProvider, ProposalPool, TriageSession

__all__ = [
    "SessionStatus",
    "SessionError",
    "StaleBatchError",
    "QueueLabelProvider",
    "PendingBatch",
    "ManagedSession",
    "SessionManager",
]


class SessionStatus(str, enum.Enum):
    """Explicit lifecycle state of a managed session."""

    STARTING = "starting"          # worker spun up, first batch not yet acquired
    AWAITING_LABELS = "awaiting_labels"  # a batch is pending; the UI must submit labels
    ADVANCING = "advancing"        # labels submitted; worker calibrating/triaging
    FINISHED = "finished"          # pool exhausted or round budget reached
    FAILED = "failed"              # worker raised; see .error -- never blocks the UI


class SessionError(RuntimeError):
    """Raised by HTTP-facing methods when the session has FAILED."""


class StaleBatchError(RuntimeError):
    """Raised when a label submission targets a batch that is no longer current.

    Guards against duplicate submits, a stale browser tab, or a second client trying to
    answer a batch that was already answered (or superseded by a later round).
    """


class QueueLabelProvider(LabelProvider):
    """A LabelProvider whose ``get_labels`` blocks until labels are pushed in.

    The session worker thread calls ``get_labels(ids)`` and blocks; it publishes the
    pending ids via ``on_request`` (so the HTTP layer can expose them) and waits on an
    event. The HTTP ``submit_labels`` handler calls :meth:`provide`, which unblocks the
    worker. This is exactly the ``CallbackLabelProvider`` contract, made asynchronous
    across threads.
    """

    def __init__(self, on_request) -> None:
        self._on_request = on_request
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._pending_ids: Optional[List[Hashable]] = None
        self._labels: Optional[Dict[Hashable, int]] = None
        self._closed = False

    def get_labels(self, proposal_ids: Sequence[Hashable]) -> Dict[Hashable, int]:
        ids = list(proposal_ids)
        if not ids:
            return {}
        with self._lock:
            self._pending_ids = ids
            self._labels = None
            self._ready.clear()
        self._on_request(ids)                 # tell the HTTP layer a batch is waiting
        self._ready.wait()                    # park until submit_labels delivers
        with self._lock:
            if self._closed:
                raise RuntimeError("session closed while awaiting labels")
            labels = dict(self._labels or {})
            self._pending_ids = None
        missing = [pid for pid in ids if pid not in labels]
        if missing:
            raise KeyError(f"missing labels for {missing[:8]}{'...' if len(missing) > 8 else ''}")
        return {pid: int(labels[pid]) for pid in ids}

    def provide(self, labels: Dict[Hashable, int]) -> None:
        """Deliver labels for the currently pending batch and unblock the worker."""
        with self._lock:
            if self._pending_ids is None:
                raise RuntimeError("no batch is awaiting labels")
            self._labels = dict(labels)
            self._ready.set()

    @property
    def pending_ids(self) -> Optional[List[Hashable]]:
        with self._lock:
            return list(self._pending_ids) if self._pending_ids is not None else None

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._ready.set()


@dataclass
class PendingBatch:
    """A batch of proposals waiting for the reviewer, with render metadata.

    ``batch_id`` is a stable, monotonic identifier for this exact pending request. The UI
    must echo it back on submit; a submission whose ``batch_id`` does not match the current
    pending batch is rejected as stale (duplicate submit / stale tab / second client).
    """

    batch_id: str
    phase: str                     # "calibration" | "review"
    round: int                     # 0 during calibration; 1-based review round otherwise
    proposal_ids: List[Hashable]
    proposals: List[dict]          # per-id: {proposal_id, image_id, file_name, bbox_xywh}


class ManagedSession:
    """One live triage session plus the machinery to drive it asynchronously.

    Lifecycle from the HTTP layer's point of view:

        m = manager.create(...)          # spins up the worker; it parks on calibration
        batch = m.wait_for_batch()       # the calibration batch (100 proposals)
        m.submit_labels({id: cls, ...})  # -> calibrate, then park on review round 1
        batch = m.wait_for_batch()       # review round 1
        m.submit_labels(...)             # -> triage + retire, park on round 2
        ...                              # until m.finished or the round budget is spent
    """

    def __init__(
        self,
        session_id: str,
        pool: ProposalPool,
        *,
        protocol: str,
        seed: int,
        batch_size: int,
        calibration_size: int,
        image_root: Optional[str],
        precomputed_density: Optional[np.ndarray],
        max_rounds: Optional[int],
    ) -> None:
        self.session_id = session_id
        self.pool = pool
        self.image_root = image_root
        self.calibration_size = int(calibration_size)
        self.max_rounds = max_rounds
        self._precomputed_density = precomputed_density

        self.provider = QueueLabelProvider(on_request=self._on_request)
        self.session = TriageSession.for_protocol(
            protocol, pool, self.provider, seed=seed, batch_size=batch_size
        )

        self._lock = threading.Lock()
        self._batch_ready = threading.Event()
        self._phase_done = threading.Event()
        self._pending: Optional[PendingBatch] = None
        self._last_round = None            # RoundInfo of the most recent completed round
        self._error: Optional[BaseException] = None
        self._status = SessionStatus.STARTING
        self._batch_seq = 0                # monotonic counter -> stable batch_id
        self._calibrated = False
        self._finished = False

        self._worker = threading.Thread(target=self._run, daemon=True, name=f"triage-{session_id}")
        self._worker.start()

    # ---- worker ------------------------------------------------------------- #
    def _phase_for(self) -> str:
        return "review" if self._calibrated else "calibration"

    def _on_request(self, ids: Sequence[Hashable]) -> None:
        """On the worker thread: a batch is now waiting for labels."""
        index = self.pool.id_to_index
        proposals = []
        for pid in ids:
            meta = self.pool.metadata[index[pid]]
            proposals.append(
                {
                    "proposal_id": pid,
                    "image_id": meta.get("image_id"),
                    "file_name": meta.get("file_name"),
                    "bbox_xywh": list(meta.get("bbox_xywh", [])),
                }
            )
        with self._lock:
            self._batch_seq += 1
            phase = self._phase_for()
            round_no = self.session.round + (0 if not self._calibrated else 1)
            batch_id = f"{self.session_id}:{phase}:{round_no}:{self._batch_seq}"
            self._pending = PendingBatch(
                batch_id=batch_id,
                phase=phase,
                round=round_no,
                proposal_ids=list(ids),
                proposals=proposals,
            )
            self._status = SessionStatus.AWAITING_LABELS
        self._batch_ready.set()

    def _run(self) -> None:
        """Run the entire session lifecycle; block in the provider whenever labels needed."""
        try:
            self.session.build_calibration_bank(
                size=self.calibration_size, precomputed_density=self._precomputed_density
            )
            with self._lock:
                self._calibrated = True
            self._phase_done.set()          # calibration complete

            round_no = 0
            while not self.session.finished:
                if self.max_rounds is not None and round_no >= self.max_rounds:
                    break
                info = self.session.step(collect_rows=True)
                round_no += 1
                with self._lock:
                    self._last_round = info
                self._phase_done.set()      # this round complete

            with self._lock:
                self._finished = True
                self._pending = None
                self._status = SessionStatus.FINISHED
        except BaseException as exc:        # ANY failure -> explicit FAILED, never hang
            with self._lock:
                self._error = exc
                self._finished = True
                self._pending = None
                self._status = SessionStatus.FAILED
        finally:
            # Wake anything waiting so no HTTP call blocks forever on a dead worker.
            self._phase_done.set()
            self._batch_ready.set()

    # ---- HTTP-facing API ---------------------------------------------------- #
    def _raise_if_failed(self) -> None:
        """If the worker crashed, raise a SessionError carrying the cause. FAILED is
        sticky -- the error is NOT consumed, so every subsequent call also sees it."""
        if self._status is SessionStatus.FAILED:
            raise SessionError(f"session failed: {self._error!r}") from self._error

    def wait_for_batch(self, timeout: float = 120.0) -> Optional[PendingBatch]:
        """Return the batch currently awaiting labels, None if finished, or raise
        SessionError if the worker failed. Never blocks past a terminal state."""
        if not self._batch_ready.wait(timeout):
            with self._lock:
                self._raise_if_failed()
                if self._finished:
                    return None
            raise TimeoutError("no batch became ready in time")
        with self._lock:
            self._raise_if_failed()
            if self._pending is None and self._finished:
                return None
            return self._pending

    def submit_labels(
        self,
        labels: Dict[Hashable, int],
        *,
        batch_id: str,
        timeout: float = 120.0,
    ):
        """Deliver labels for the pending batch and wait for that phase to complete.

        ``batch_id`` (required) must equal the current pending batch's id; otherwise a
        :class:`StaleBatchError` is raised and NO labels are consumed -- this makes submits
        idempotent/stale-safe against duplicate posts, a stale tab, or a second client.
        Returns the completed round's ``RoundInfo`` (``None`` for the calibration phase).

        Validation AND consumption of the pending batch happen in a single critical
        section, so two concurrent submissions for the same batch cannot both pass the
        stale check -- exactly one wins and consumes the batch; the loser raises
        :class:`StaleBatchError`.
        """
        with self._lock:
            self._raise_if_failed()
            if self._pending is None:
                raise StaleBatchError("no batch is awaiting labels (already answered or finished)")
            if batch_id != self._pending.batch_id:
                raise StaleBatchError(
                    f"stale batch_id {batch_id!r}; current is {self._pending.batch_id!r}"
                )
            pending_ids = list(self._pending.proposal_ids)
            was_calibration = self._pending.phase == "calibration"
            missing = [pid for pid in pending_ids if pid not in labels]
            if missing:
                # Reject WITHOUT consuming the batch: an incomplete submission leaves the
                # batch pending so a corrected resubmission can still answer it.
                raise KeyError(f"labels missing for pending proposals: {missing[:8]}")
            # Atomically consume the batch under the same lock that validated it, so a
            # concurrent duplicate submit finds either no pending batch or a mismatched id.
            resolved_labels = {pid: int(labels[pid]) for pid in pending_ids}
            self._pending = None
            self._status = SessionStatus.ADVANCING
            self._phase_done.clear()
            self._batch_ready.clear()

        # Outside the lock: hand labels to the parked worker (which needs the lock) and
        # wait for it to finish the phase (waiting under the lock would deadlock).
        self.provider.provide(resolved_labels)

        if not self._phase_done.wait(timeout):
            with self._lock:
                self._raise_if_failed()
            raise TimeoutError("phase did not complete after labels were submitted")
        with self._lock:
            self._raise_if_failed()
            return None if was_calibration else self._last_round

    # ---- introspection ------------------------------------------------------ #
    @property
    def status(self) -> SessionStatus:
        with self._lock:
            return self._status

    @property
    def error(self) -> Optional[BaseException]:
        with self._lock:
            return self._error

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._status is SessionStatus.FINISHED

    @property
    def failed(self) -> bool:
        with self._lock:
            return self._status is SessionStatus.FAILED

    @property
    def calibrated(self) -> bool:
        with self._lock:
            return self._calibrated

    def state(self) -> dict:
        with self._lock:
            pending = self._pending
            status = self._status
            error = self._error
        return {
            "session_id": self.session_id,
            "status": status.value,
            # Only the exception TYPE is exposed to clients; the full message (which may
            # contain OSC filesystem paths) is available server-side via ``.error``.
            "error": None if error is None else type(error).__name__,
            "calibrated": self._calibrated,
            "rule": self.session.rule.as_dict() if self.session.rule else None,
            "round": self.session.round,
            "cumulative_reviews": self.session.cumulative_reviews,
            "remaining": int(self.session._available.size),
            "finished": status is SessionStatus.FINISHED,
            "failed": status is SessionStatus.FAILED,
            "pending_batch": None if pending is None else {
                "batch_id": pending.batch_id,
                "phase": pending.phase, "round": pending.round,
                "size": len(pending.proposal_ids),
            },
        }

    def close(self) -> None:
        self.provider.close()
        with self._lock:
            if self._status not in (SessionStatus.FINISHED, SessionStatus.FAILED):
                self._finished = True
                self._status = SessionStatus.FINISHED


class SessionManager:
    """Creates and tracks :class:`ManagedSession` instances by id (in-memory)."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ManagedSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        pool: ProposalPool,
        *,
        protocol: str = "split_no_veto",
        seed: int = 0,
        batch_size: int = 50,
        calibration_size: int = 100,
        image_root: Optional[str] = None,
        precomputed_density: Optional[np.ndarray] = None,
        max_rounds: Optional[int] = None,
    ) -> ManagedSession:
        session_id = uuid.uuid4().hex
        managed = ManagedSession(
            session_id,
            pool,
            protocol=protocol,
            seed=seed,
            batch_size=batch_size,
            calibration_size=calibration_size,
            image_root=image_root,
            precomputed_density=precomputed_density,
            max_rounds=max_rounds,
        )
        with self._lock:
            self._sessions[session_id] = managed
        return managed

    def get(self, session_id: str) -> ManagedSession:
        with self._lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            return self._sessions[session_id]

    def remove(self, session_id: str) -> None:
        with self._lock:
            managed = self._sessions.pop(session_id, None)
        if managed is not None:
            managed.close()
