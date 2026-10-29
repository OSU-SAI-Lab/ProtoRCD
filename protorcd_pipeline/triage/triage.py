"""Guarded Resolve / Clear / Defer triage decision (stage 3, part 2).

This is the canonical Prototype-Triage rule from the paper. It is **asymmetric** by
design and uses two representations:

  * foreground evidence -- BioCLIP-2 embeddings (``fg_x``): kNN species votes.
  * background evidence -- DINOv3 embeddings   (``bg_x``): kNN "is-background" votes.

For each query proposal, given the *reviewed support set* (embeddings + human labels,
background == class 0):

  Clear   : DINOv3 gives >= tau_bg background votes AND foreground does not strongly
            object (max FG vote < tau_veto, the "guarded veto").
  Resolve : the winning FG species has >= tau_fg votes, is the unique top class, has
            >= 2 reviewed exemplars, and its best support similarity clears a
            per-class floor (the support quantile). Unless foreground_first, a proposal
            already eligible for Clear is not Resolved.
  Defer   : everything else -> sent to a human.

The decision reads ONLY the reviewed support labels and the two embeddings. Ground
truth is never consulted here; ``true_y`` is optional and used only to attach
diagnostic ``gt_*`` fields when labels happen to be known (offline/eval runs).

Ported logic-for-logic from ``evaluate_two_stage_calibrated_guarded_initial100.py``
(``guarded_triage_with_diagnostics`` / ``class_support_thresholds``) and
``evaluate_active_triage_features.py``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "TriageRule",
    "class_support_thresholds",
    "topk_indices_sims",
    "guarded_triage",
]


@dataclass(frozen=True)
class TriageRule:
    """Calibrated integer-vote thresholds for the guarded triage rule."""

    k: int
    tau_fg: int
    tau_bg: int
    tau_veto: int
    support_quantile: float = 0.5
    foreground_first: bool = False

    def as_dict(self) -> Dict[str, float | int | bool]:
        return {
            "k": int(self.k),
            "tau_fg": int(self.tau_fg),
            "tau_bg": int(self.tau_bg),
            "tau_veto": int(self.tau_veto),
            "support_quantile": float(self.support_quantile),
            "foreground_first": bool(self.foreground_first),
        }


def class_support_thresholds(
    support_x: np.ndarray, support_y: np.ndarray, quantile: float
) -> Dict[int, float]:
    """Per-class similarity floor: the ``quantile`` of each class's within-class
    nearest-neighbor similarities. Classes with < 2 exemplars get an infinite floor
    (never auto-resolvable)."""
    out: Dict[int, float] = {}
    for cls in sorted(set(int(v) for v in support_y.tolist())):
        idx = np.flatnonzero(support_y == cls)
        if idx.size < 2:
            out[cls] = float("inf")
            continue
        sims = support_x[idx] @ support_x[idx].T
        np.fill_diagonal(sims, -np.inf)
        out[cls] = float(np.quantile(np.max(sims, axis=1), np.clip(quantile, 0.0, 1.0)))
    return out


def topk_indices_sims(
    support_x: np.ndarray,
    query_x: np.ndarray,
    k: int,
    chunk_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k support indices and cosine sims for each query row (descending sim)."""
    k = min(max(1, int(k)), support_x.shape[0])
    top_idx_all = np.empty((query_x.shape[0], k), dtype=np.int64)
    top_sims_all = np.empty((query_x.shape[0], k), dtype=np.float32)
    for start in range(0, query_x.shape[0], chunk_size):
        stop = min(start + chunk_size, query_x.shape[0])
        sims = query_x[start:stop] @ support_x.T
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        local = np.take_along_axis(sims, idx, axis=1)
        order = np.argsort(-local, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        top_sims = np.take_along_axis(sims, idx, axis=1)
        top_idx_all[start:stop] = idx
        top_sims_all[start:stop] = top_sims
    return top_idx_all, top_sims_all


def guarded_triage(
    support_bg_x: np.ndarray,
    support_fg_x: np.ndarray,
    support_y: np.ndarray,
    query_bg_x: np.ndarray,
    query_fg_x: np.ndarray,
    rule: TriageRule,
    *,
    query_global_idx: Optional[np.ndarray] = None,
    true_y: Optional[np.ndarray] = None,
    support_global_idx: Optional[np.ndarray] = None,
    collect_rows: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, object]]]:
    """Run the guarded Resolve/Clear/Defer rule over a query pool.

    Parameters
    ----------
    support_bg_x, support_fg_x : (S, D) reviewed-support DINOv3 / BioCLIP-2 embeddings.
    support_y                  : (S,) reviewed-support labels (0 == background).
    query_bg_x, query_fg_x     : (Q, D) query DINOv3 / BioCLIP-2 embeddings.
    rule                       : calibrated :class:`TriageRule`.
    query_global_idx           : optional (Q,) stable proposal indices/ids for logging.
    true_y                     : optional (Q,) ground-truth labels for DIAGNOSTICS ONLY
                                 (offline/eval); never used in the decision.
    support_global_idx         : optional (S,) stable ids for the support prototypes
                                 (attached to per-row top-k diagnostics).
    collect_rows               : if False, skip building the per-proposal diagnostic
                                 dicts (faster; returns an empty ``rows`` list).

    Returns
    -------
    pred      : (Q,) predicted class for Resolve rows (0 for Clear/undecided).
    auto      : (Q,) bool -- True where the rule auto-decided (Resolve or Clear).
    decisions : list[str] of length Q, each "resolve" | "clear" | "defer".
    rows      : list[dict] per-proposal diagnostics (empty if collect_rows=False).
    """
    k = rule.k
    tau_fg, tau_bg, tau_veto = rule.tau_fg, rule.tau_bg, rule.tau_veto
    foreground_first = rule.foreground_first

    q_n = query_bg_x.shape[0]
    k_eff = min(max(1, int(k)), support_y.shape[0])
    bg_top_idx, bg_top_sims = topk_indices_sims(support_bg_x, query_bg_x, k_eff)
    fg_top_idx, fg_top_sims = topk_indices_sims(support_fg_x, query_fg_x, k_eff)
    bg_top_labels = support_y[bg_top_idx]
    fg_top_labels = support_y[fg_top_idx]
    thresholds = class_support_thresholds(support_fg_x, support_y, rule.support_quantile)
    all_support_counts = Counter(int(v) for v in support_y.tolist())

    pred = np.zeros((q_n,), dtype=np.int64)
    auto = np.zeros((q_n,), dtype=bool)
    decisions: List[str] = []
    rows: List[Dict[str, object]] = []

    if query_global_idx is None:
        query_global_idx = np.arange(q_n, dtype=np.int64)

    for row in range(q_n):
        labels_fg = [int(v) for v in fg_top_labels[row].tolist() if int(v) != 0]
        if labels_fg:
            counts = Counter(labels_fg)
            max_fg_vote = max(int(v) for v in counts.values())
            winners = sorted(int(cls) for cls, votes in counts.items() if int(votes) == max_fg_vote)
            winning_cls = int(winners[0])
            unique = len(winners) == 1
        else:
            max_fg_vote = 0
            winning_cls = 0
            unique = False

        bg_vote_count = int(np.sum(bg_top_labels[row] == 0))
        veto_pass = bool(max_fg_vote >= int(tau_veto))
        bg_condition = bool(bg_vote_count >= int(tau_bg))
        candidate_clear = bg_condition
        clear = bool(candidate_clear and not veto_pass)

        cls_cols = (
            np.flatnonzero(support_y == winning_cls)
            if winning_cls != 0
            else np.asarray([], dtype=np.int64)
        )
        floor_value = thresholds.get(int(winning_cls), float("inf"))
        if cls_cols.size:
            winning_sim = float(np.max(query_fg_x[row] @ support_fg_x[cls_cols].T))
        else:
            winning_sim = float("nan")
        floor_pass = bool(np.isfinite(floor_value) and winning_sim >= float(floor_value))
        min_support_pass = bool(int(all_support_counts.get(int(winning_cls), 0)) >= 2)
        fg_condition = bool(
            max_fg_vote >= int(tau_fg) and unique and min_support_pass and floor_pass
        )

        if foreground_first and fg_condition:
            auto[row] = True
            pred[row] = int(winning_cls)
            decision = "resolve"
        elif clear:
            auto[row] = True
            pred[row] = 0
            decision = "clear"
        elif fg_condition:
            auto[row] = True
            pred[row] = int(winning_cls)
            decision = "resolve"
        else:
            pred[row] = int(winning_cls) if winning_cls != 0 else 0
            decision = "defer"
        decisions.append(decision)

        if not collect_rows:
            continue

        qidx = int(query_global_idx[row])
        r: Dict[str, object] = {
            "proposal_index": qidx,
            "decision": decision,
            "predicted_fg_class": int(pred[row]) if decision == "resolve" else "",
            "auto": int(auto[row]),
            "fg_winning_class": int(winning_cls) if winning_cls != 0 else "",
            "fg_vote_count": int(max_fg_vote),
            "bg_vote_count": int(bg_vote_count),
            "fg_condition_pass": int(fg_condition),
            "bg_condition_pass": int(bg_condition),
            "candidate_clear": int(candidate_clear),
            "veto_pass": int(veto_pass),
            "similarity_floor_value": float(floor_value) if np.isfinite(floor_value) else "",
            "winning_class_similarity": float(winning_sim) if np.isfinite(winning_sim) else "",
            "similarity_floor_pass": int(floor_pass),
            "minimum_support_pass": int(min_support_pass),
            "winning_class_support_count": int(all_support_counts.get(int(winning_cls), 0)),
            "k": int(k_eff),
            "tau_fg": int(tau_fg),
            "tau_bg": int(tau_bg),
            "tau_veto": int(tau_veto),
        }
        if support_global_idx is not None:
            r["bioclip_topk_prototype_indices"] = [int(v) for v in support_global_idx[fg_top_idx[row]].tolist()]
            r["dinov3_topk_prototype_indices"] = [int(v) for v in support_global_idx[bg_top_idx[row]].tolist()]
        r["bioclip_topk_labels"] = [int(v) for v in fg_top_labels[row].tolist()]
        r["bioclip_topk_similarities"] = [round(float(v), 6) for v in fg_top_sims[row].tolist()]
        r["dinov3_topk_labels"] = [int(v) for v in bg_top_labels[row].tolist()]
        r["dinov3_topk_similarities"] = [round(float(v), 6) for v in bg_top_sims[row].tolist()]
        if true_y is not None:
            r["gt_label"] = int(true_y[row])
            r["gt_is_foreground"] = int(true_y[row] != 0)
        rows.append(r)

    return pred, auto, decisions, rows
