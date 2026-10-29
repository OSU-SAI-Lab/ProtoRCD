"""Active-triage session -- the round loop a UI drives (stage 3, orchestration).

This ties the stage-3 pieces together into the loop the paper runs, but decoupled from
ground truth: labels arrive through a :class:`~protorcd_pipeline.triage.labels.LabelProvider`
(a UI callback, a JSON exchange, or -- offline only -- a GT provider).

Lifecycle::

    session = TriageSession.for_protocol(
        "split_no_veto", pool, provider, seed=3)

    # 1. Build the calibration bank (held out from acquisition + triage forever) and
    #    calibrate the guarded rule on its reviewed labels.
    session.build_calibration_bank(size=100)

    # 2. Iterate review rounds. Each round: acquire a batch -> ask the provider for
    #    labels -> add them to the support set -> triage the rest and RETIRE the
    #    auto-decided (Resolve or Clear) proposals from the pool. A UI calls step()
    #    once per round; the auto-decisions come back on the returned ``RoundInfo``.
    while not session.finished:
        info = session.step()               # acquire + label + triage + retire
        # info.triage holds this round's Resolve/Clear/Defer over the remaining pool

State machine (matches the research reference exactly)
------------------------------------------------------
The kept pool (everything except the calibration bank) is partitioned every round into
three disjoint sets whose union is the whole kept pool -- the invariant the research
asserts with ``assert_round_accounting``:

    reviewed (support)  U  auto-decided (retired)  U  remaining  ==  kept pool

* **reviewed** -- acquired for human review; only these enter the prototype/support bank.
* **auto-decided** -- Resolved or Cleared by the calibrated rule; they leave the pool
  permanently and are never re-acquired. Their predictions are NOT supervision.
* **remaining** -- deferred; the only proposals eligible for the next round.

Ported orchestration-for-orchestration from
``evaluate_final_shared_random_calibrated_zero_start_protocol_ablation.py`` (the round
loop at "for round_idx in range(...)"), decoupled from ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .acquisition import acquire, compute_density
from .calibration import DEFAULT_K_VALUES, calibrate_rule
from .labels import LabelProvider
from .pool import ProposalPool
from .triage import TriageRule, guarded_triage

__all__ = ["TriageSession", "RoundInfo", "TriageResult", "PROTOCOL_PRESETS"]


# Canonical protocol presets. ``split_no_veto`` is the paper's headline configuration:
# Resolve strictly before Clear (``foreground_first``), the BioCLIP veto disabled, and
# a per-k semantic-supermajority calibration grid. These mirror the authoritative launch
# config ``sbatch_wacv5_lote_train_test_main.sh`` and the research
# ``calibrate_rule_for_protocol`` path.
PROTOCOL_PRESETS: Dict[str, Dict[str, object]] = {
    "split_no_veto": {
        "foreground_first": True,
        "calibration_mode": "semantic_supermajority",
        "support_quantile": 0.0,
        "k_values": (5, 7, 9, 11),
        "method": "density_balanced_kcenter",
        "acquire_on": "foreground",
        "density_mode": "full_knn",
        "density_k": 30,
        "density_scope": "initial",
        "batch_size": 50,
    },
}


@dataclass
class TriageResult:
    """Auto-decisions over a query pool from one call to the guarded rule."""

    proposal_ids: List
    decisions: List[str]                  # "resolve" | "clear" | "defer"
    predicted_class: np.ndarray           # 0 for clear/defer
    auto: np.ndarray                      # bool -- True where Resolved or Cleared
    rows: List[dict] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        c = {"resolve": 0, "clear": 0, "defer": 0}
        for d in self.decisions:
            c[d] += 1
        return c


@dataclass
class RoundInfo:
    round: int
    acquired_ids: List
    acquired_labels: Dict
    cumulative_reviews: int
    remaining: int
    triage: Optional[TriageResult] = None   # this round's decisions over the pool
    retired: int = 0                        # count of proposals auto-decided this round


class TriageSession:
    """Drive the paper's active Resolve/Clear/Defer loop, ground-truth-decoupled.

    The generic ``__init__`` keeps neutral defaults (``foreground_first=False`` etc.).
    For an exact reproduction of a paper protocol use :meth:`for_protocol`, which applies
    the calibrated configuration in :data:`PROTOCOL_PRESETS`.
    """

    def __init__(
        self,
        pool: ProposalPool,
        label_provider: LabelProvider,
        *,
        method: str = "density_balanced_kcenter",
        batch_size: int = 50,
        seed: int = 0,
        acquire_on: str = "foreground",       # which embedding drives acquisition
        density_mode: str = "full_knn",
        density_k: int = 20,
        density_anchor_count: int = 4096,
        density_scope: str = "round",         # "initial" (paper) or "round"
        k_values: Sequence[int] = DEFAULT_K_VALUES,
        support_quantile: float = 0.5,
        calibration_mode: str = "free_thresholds",
        foreground_first: bool = False,
    ) -> None:
        self.pool = pool
        self.provider = label_provider
        self.method = method
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.acquire_on = acquire_on
        self.density_mode = density_mode
        self.density_k = int(density_k)
        self.density_anchor_count = int(density_anchor_count)
        self.density_scope = density_scope
        self.k_values = tuple(int(v) for v in k_values)
        self.support_quantile = float(support_quantile)
        self.calibration_mode = calibration_mode
        self.foreground_first = bool(foreground_first)

        n = pool.size
        # ``kept_source`` maps compacted (kept-space) rows -> original pool rows. It is
        # the identity until build_calibration_bank() removes the calibration bank, after
        # which every partition below lives in kept-space -- matching the research code,
        # whose decision-cache indices are kept-space (calibration removed).
        self._kept_source = np.arange(n, dtype=np.int64)
        self._calibration_idx = np.asarray([], dtype=np.int64)   # original pool rows
        self._support_idx: List[int] = []          # kept-space, in acquisition order
        self._support_labels: List[int] = []        # aligned to _support_idx
        # available = kept-space rows not yet reviewed and not yet auto-decided
        self._available = np.arange(n, dtype=np.int64)
        self._auto_idx: List[int] = []              # kept-space, retired (Resolve/Clear)
        self._auto_pred: List[int] = []             # aligned to _auto_idx

        self.rule: Optional[TriageRule] = None
        self.calibration_report: Optional[dict] = None
        self.round = 0
        self._initial_density: Optional[np.ndarray] = None   # kept-space, scope="initial"

        # Acquisition representation, in ORIGINAL pool space; compacted to kept-space in
        # build_calibration_bank().
        self._acquire_x = pool.fg_x if acquire_on == "foreground" else pool.bg_x

    # ---- construction ------------------------------------------------------- #
    @classmethod
    def for_protocol(
        cls,
        protocol: str,
        pool: ProposalPool,
        label_provider: LabelProvider,
        *,
        seed: int = 0,
        **overrides: object,
    ) -> "TriageSession":
        """Build a session pre-configured for a named paper protocol.

        ``overrides`` win over the preset (e.g. a different ``batch_size``). This is the
        entry point for reproducing published results; the plain constructor is for
        custom/experimental configurations.
        """
        if protocol not in PROTOCOL_PRESETS:
            raise ValueError(
                f"Unknown protocol {protocol!r}; known: {sorted(PROTOCOL_PRESETS)}"
            )
        cfg: Dict[str, object] = dict(PROTOCOL_PRESETS[protocol])
        cfg.update(overrides)
        return cls(pool, label_provider, seed=seed, **cfg)  # type: ignore[arg-type]

    # ---- kept-space bookkeeping -------------------------------------------- #
    def _to_pool_idx(self, kept_idx: np.ndarray) -> np.ndarray:
        """Map kept-space rows back to original pool rows."""
        return self._kept_source[np.asarray(kept_idx, dtype=np.int64)]

    # ---- calibration -------------------------------------------------------- #
    def build_calibration_bank(
        self, size: int = 100, *, precomputed_density: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Draw a fixed random calibration bank, review it, calibrate, and compact.

        The draw depends only on ``seed`` (matching the paper's per-seed bank). The bank
        is removed from the pool permanently; the remaining rows are compacted into
        kept-space (``arange(N - size)``) so indices line up with the research decision
        cache. Calibration labels come through the same LabelProvider as everything else.

        ``precomputed_density`` (optional, ``density_scope="initial"`` only) supplies the
        kept-space local density directly instead of recomputing it. It must be aligned to
        kept-space natural order -- i.e. length ``N - size`` and ordered like
        ``kept_source``. Use it to consume a frozen density array (a large float32 kNN
        matmul is not bit-for-bit reproducible across BLAS/hardware, so pinning the exact
        density is the only way to reproduce a run's tie-breaking on dense pools).

        Returns the calibration indices in ORIGINAL pool space.
        """
        if self._support_idx or self.round:
            raise RuntimeError("build_calibration_bank must be called before any review round")
        n = self.pool.size
        size = min(int(size), n)
        # A SINGLE rng draws the calibration bank AND (below) the kept-pool review order,
        # in that order -- reproducing the research stream exactly (one
        # ``np.random.default_rng(seed)`` used for both draws).
        rng = np.random.default_rng(self.seed)
        bank = rng.choice(n, size=size, replace=False).astype(np.int64)
        self._calibration_idx = np.sort(bank)

        # Review the bank (original-space ids) and calibrate on its labels.
        ids = [self.pool.proposal_ids[i] for i in self._calibration_idx.tolist()]
        labels = self.provider.get_labels(ids)
        y_full = np.zeros((n,), dtype=np.int64)
        for i, pid in zip(self._calibration_idx.tolist(), ids):
            y_full[i] = int(labels[pid])

        rule, best, all_rows = calibrate_rule(
            self.pool.bg_x, self.pool.fg_x, y_full, self._calibration_idx,
            k_values=self.k_values,
            support_quantile=self.support_quantile,
            calibration_mode=self.calibration_mode,
            foreground_first=self.foreground_first,
        )
        self.rule = rule
        self._calibration_labels = {int(i): int(y_full[i]) for i in self._calibration_idx.tolist()}
        self.calibration_report = {
            "rule": rule.as_dict(),
            "loo": best,
            "bank_size": int(size),
            "n_candidates": len(all_rows),
        }

        # Compact to kept-space: drop calibration rows, renumber the rest to arange(N-size).
        keep = np.ones((n,), dtype=bool)
        keep[self._calibration_idx] = False
        self._kept_source = np.flatnonzero(keep)
        self._acquire_x = self._acquire_x[self._kept_source]
        # Review order: permute kept-space from the SAME rng (after the calibration draw),
        # matching research ``review_idx = rng.permutation(arange(N - size))``. Acquisition
        # then picks positions into this order; farthest-point selection is
        # order-insensitive except for exact argmax ties, so this keeps ties resolved the
        # same way the research code resolves them.
        self._available = rng.permutation(
            np.arange(self._kept_source.size, dtype=np.int64)
        )

        # Scope="initial": density is computed ONCE over the full kept pool now, then
        # each round slices ``density[available]`` -- never recomputed on the shrinking
        # pool (matching the research density_scope="initial" path).
        if self.density_scope == "initial" and self.method in (
            "density_kcenter", "density_balanced_kcenter"
        ):
            if precomputed_density is not None:
                dens = np.asarray(precomputed_density, dtype=np.float32)
                if dens.shape[0] != self._kept_source.size:
                    raise ValueError(
                        f"precomputed_density length {dens.shape[0]} != kept pool "
                        f"{self._kept_source.size}"
                    )
                self._initial_density = dens
            else:
                self._initial_density = compute_density(
                    self._acquire_x, self.density_mode, self.density_k,
                    self.density_anchor_count, seed=self.seed,
                )
        elif precomputed_density is not None:
            raise ValueError(
                "precomputed_density is only valid with density_scope='initial' and a "
                "density-based method"
            )
        return self._calibration_idx

    # ---- acquisition round -------------------------------------------------- #
    @property
    def finished(self) -> bool:
        return self._available.size == 0

    @property
    def cumulative_reviews(self) -> int:
        return len(self._support_idx)

    def _round_density(self, avail: np.ndarray) -> np.ndarray:
        """Local density for the current available pool for the acquisition policy."""
        if self.method not in ("density_kcenter", "density_balanced_kcenter"):
            return np.ones((avail.size,), dtype=np.float32)
        if self.density_scope == "initial":
            assert self._initial_density is not None
            return self._initial_density[avail]
        # scope="round": recompute on the shrinking pool with a per-round seed.
        return compute_density(
            self._acquire_x[avail], self.density_mode, self.density_k,
            self.density_anchor_count, seed=self.seed + 1009 * self.round,
        )

    def step(self, *, collect_rows: bool = True) -> RoundInfo:
        """Run one full round: acquire a batch, label it, triage the rest, retire autos.

        This is one atomic state transition matching the research round loop:
          1. acquire ``batch_size`` from the available pool and get human labels;
          2. add them to the support set (the only supervision);
          3. run the calibrated rule over the pool MINUS this round's acquisitions;
          4. retire every auto-decided (Resolve or Clear) proposal from the pool so it is
             never re-acquired; only deferred proposals survive to the next round.

        The reviewed/auto/remaining partition invariant is asserted every round.
        """
        if self.finished:
            raise RuntimeError("No proposals left to acquire")
        if self.rule is None:
            raise RuntimeError("Call build_calibration_bank() before step()")
        self.round += 1
        avail = self._available
        x = self._acquire_x[avail]

        # 1. acquire -------------------------------------------------------- #
        density = self._round_density(avail)
        local = acquire(self.method, x, self.batch_size, density, seed=self.seed + 1009 * self.round)
        chosen = avail[local]                                  # kept-space rows

        pool_ids = [self.pool.proposal_ids[i] for i in self._to_pool_idx(chosen).tolist()]
        labels = self.provider.get_labels(pool_ids)
        for i, pid in zip(chosen.tolist(), pool_ids):
            self._support_idx.append(int(i))
            self._support_labels.append(int(labels[pid]))

        # candidate pool = available minus this round's freshly acquired rows.
        keep_avail = np.ones((avail.size,), dtype=bool)
        keep_avail[local] = False
        candidate = avail[keep_avail]

        # 2/3. triage the candidate pool with the up-to-date support set ------ #
        result = self._triage_indices(candidate, collect_rows=collect_rows)

        # 4. retire the auto-decided; only deferred remain ------------------- #
        auto_mask = result.auto
        round_auto = candidate[auto_mask]
        self._auto_idx.extend(int(v) for v in round_auto.tolist())
        self._auto_pred.extend(int(v) for v in result.predicted_class[auto_mask].tolist())
        self._available = candidate[~auto_mask]
        self._assert_round_accounting()

        return RoundInfo(
            round=self.round,
            acquired_ids=pool_ids,
            acquired_labels={pid: int(labels[pid]) for pid in pool_ids},
            cumulative_reviews=self.cumulative_reviews,
            remaining=int(self._available.size),
            triage=result,
            retired=int(round_auto.size),
        )

    def _assert_round_accounting(self) -> None:
        """Enforce reviewed U auto U remaining == kept pool, all disjoint and unique."""
        selected = np.asarray(self._support_idx, dtype=np.int64)
        auto = np.asarray(self._auto_idx, dtype=np.int64)
        review = self._available
        parts = [("selected", selected), ("auto", auto), ("remaining", review)]
        for name, arr in parts:
            if arr.size and np.unique(arr).size != arr.size:
                raise AssertionError(f"round {self.round}: duplicate {name} indices")
        for (n1, a1), (n2, a2) in ((parts[0], parts[1]), (parts[0], parts[2]), (parts[1], parts[2])):
            if a1.size and a2.size and np.intersect1d(a1, a2).size:
                raise AssertionError(f"round {self.round}: proposal is both {n1} and {n2}")
        total = int(selected.size + auto.size + review.size)
        if total != int(self._kept_source.size):
            raise AssertionError(
                f"round {self.round}: accounting total {total} != kept pool {self._kept_source.size}"
            )

    # ---- triage ------------------------------------------------------------- #
    def _triage_indices(self, query_idx: np.ndarray, *, collect_rows: bool) -> TriageResult:
        """Run the calibrated guarded rule over kept-space ``query_idx``."""
        query_idx = np.asarray(query_idx, dtype=np.int64)
        if query_idx.size == 0 or not self._support_idx:
            return TriageResult([], [], np.asarray([], np.int64), np.asarray([], bool), [])

        support_idx = np.asarray(self._support_idx, dtype=np.int64)          # kept-space
        support_y = np.asarray(self._support_labels, dtype=np.int64)
        support_pool = self._to_pool_idx(support_idx)
        query_pool = self._to_pool_idx(query_idx)

        true_y = self.pool.gt_labels[query_pool] if self.pool.gt_labels is not None else None

        pred, auto, decisions, rows = guarded_triage(
            support_bg_x=self.pool.bg_x[support_pool],
            support_fg_x=self.pool.fg_x[support_pool],
            support_y=support_y,
            query_bg_x=self.pool.bg_x[query_pool],
            query_fg_x=self.pool.fg_x[query_pool],
            rule=self.rule,
            query_global_idx=query_idx,
            true_y=true_y,
            support_global_idx=support_idx,
            collect_rows=collect_rows,
        )
        ids = [self.pool.proposal_ids[i] for i in query_pool.tolist()]
        if collect_rows:
            for r, pid in zip(rows, ids):
                r["proposal_id"] = pid
        return TriageResult(ids, decisions, pred, auto, rows)

    def triage(self, *, on: str = "remaining", collect_rows: bool = True) -> TriageResult:
        """Run the calibrated guarded rule over a query set (read-only; no retirement).

        ``on="remaining"`` (default) triages the current deferred pool -- the same set
        ``step()`` will retire from next round. ``on="all"`` triages every kept
        (non-calibration) proposal, including reviewed ones, for full-pool diagnostics.

        Note: retirement happens inside :meth:`step`. Calling ``triage()`` between rounds
        is a non-mutating snapshot; it does not remove anything from the pool.
        """
        if self.rule is None:
            raise RuntimeError("Call build_calibration_bank() before triage()")
        if not self._support_idx:
            raise RuntimeError("No reviewed support yet; call step() at least once")
        if on == "remaining":
            query_idx = self._available
        elif on == "all":
            query_idx = np.arange(self._kept_source.size, dtype=np.int64)
        else:
            raise ValueError(f"Unknown triage target: {on!r}")
        return self._triage_indices(query_idx, collect_rows=collect_rows)

    # ---- introspection ------------------------------------------------------ #
    @property
    def support_indices(self) -> np.ndarray:
        """Reviewed-support rows in ORIGINAL pool space."""
        return self._to_pool_idx(np.asarray(self._support_idx, dtype=np.int64))

    @property
    def support_indices_kept(self) -> np.ndarray:
        """Reviewed-support rows in kept-space (align with the research decision cache)."""
        return np.asarray(self._support_idx, dtype=np.int64)

    @property
    def support_labels(self) -> np.ndarray:
        return np.asarray(self._support_labels, dtype=np.int64)

    @property
    def auto_indices_kept(self) -> np.ndarray:
        """Auto-decided (retired) rows in kept-space, in retirement order."""
        return np.asarray(self._auto_idx, dtype=np.int64)

    @property
    def auto_predictions(self) -> np.ndarray:
        """Predicted class for each retired row (0 == Clear), aligned to auto_indices."""
        return np.asarray(self._auto_pred, dtype=np.int64)

    @property
    def calibration_indices(self) -> np.ndarray:
        """Calibration-bank rows in ORIGINAL pool space."""
        return self._calibration_idx

    @property
    def kept_source(self) -> np.ndarray:
        """Map from kept-space rows to original pool rows."""
        return self._kept_source
