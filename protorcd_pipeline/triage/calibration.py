"""Leave-one-out calibration of the guarded triage rule (stage 3, part 3).

Given a small *reviewed* calibration bank (embeddings for both representations plus
human labels, background == 0), pick the integer-vote thresholds
``(k, tau_fg, tau_bg, tau_veto)`` by leave-one-out (LOO) scoring on the bank and a
safety-first ranking.

The labels used here are the human review labels of the calibration proposals -- NOT
ground-truth box matches. In offline/eval mode those labels happen to come from a GT
provider; in UI mode they come from the reviewer. Either way the calibrator only ever
sees the reviewed bank, never the query pool or its ground truth.

Two calibration modes, matching the two research entry points:

* ``semantic_supermajority`` (the paper's ``split_no_veto`` protocol) -- exactly one
  candidate per ``k``: ``tau_fg = tau_bg = floor(2k/3) + 1`` (a two-thirds
  supermajority) and the veto disabled (``tau_veto = k + 1``). Ported from
  ``evaluate_final_shared_random_calibrated_zero_start_protocol_ablation``:
  ``candidate_thresholds`` / ``calibrate_rule_for_protocol``.
* ``free_thresholds`` -- the guarded-init grid: sweep ``tau_fg, tau_bg`` over ``2..k``
  and ``tau_veto`` over ``2..tau_fg``.

Ranking (safety-first), lower-is-better unless negated -- ported verbatim from
``evaluate_two_stage_calibrated_guarded_initial100.calibrate_rule`` (identical to the
final-run ``calibration_sort_key``):
    1. fewest FG->BG errors            (never silently drop an animal)
    2. lowest FG->BG rate
    3. highest correct-auto %          (negated)
    4. lowest deferred %               (automate more)
    5. smaller k, then higher tau_fg/tau_bg/tau_veto (conservative tie-breaks)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .triage import TriageRule, class_support_thresholds

__all__ = [
    "candidate_thresholds",
    "loo_score",
    "calibrate_rule",
    "DEFAULT_K_VALUES",
    "CALIBRATION_MODES",
]

DEFAULT_K_VALUES: Tuple[int, ...] = (3, 5, 7, 9)
CALIBRATION_MODES: Tuple[str, ...] = ("semantic_supermajority", "free_thresholds")


def candidate_thresholds(
    k: int, mode: str = "free_thresholds"
) -> Iterable[Tuple[int, int, int]]:
    """Enumerate candidate (tau_fg, tau_bg, tau_veto) vote-count triples for a given k.

    ``semantic_supermajority`` yields the single canonical triple
    ``(floor(2k/3)+1, floor(2k/3)+1, k+1)`` -- veto disabled.

    ``free_thresholds`` sweeps tau_fg, tau_bg over ``2..k`` and tau_veto over
    ``2..tau_fg``.
    """
    k = int(k)
    if mode == "semantic_supermajority":
        supermajority = int(math.floor(2 * k / 3) + 1)
        yield supermajority, supermajority, k + 1  # tau_veto = k + 1 disables the veto
        return
    if mode == "free_thresholds":
        for tau_fg in range(2, k + 1):
            for tau_bg in range(2, k + 1):
                for tau_veto in range(2, tau_fg + 1):
                    yield int(tau_fg), int(tau_bg), int(tau_veto)
        return
    raise ValueError(
        f"Unsupported calibration mode: {mode!r}; known: {CALIBRATION_MODES}"
    )


def _topk_labels(query_row: np.ndarray, support_x: np.ndarray, support_y: np.ndarray, k: int) -> np.ndarray:
    k = min(max(1, int(k)), support_x.shape[0])
    sims = query_row @ support_x.T
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return support_y[idx]


def loo_score(
    bg_x: np.ndarray,
    fg_x: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    k: int,
    tau_fg: int,
    tau_bg: int,
    tau_veto: int,
    support_quantile: float,
    foreground_first: bool = False,
) -> Dict[str, float | int]:
    """Leave-one-out score of one threshold setting over the calibration bank.

    ``bg_x``/``fg_x`` are DINOv3 / BioCLIP-2 embeddings for the WHOLE pool; ``selected``
    are the (global) indices of the calibration bank; ``y`` the labels (0 == bg).
    """
    selected = np.asarray(selected, dtype=np.int64)
    n = int(selected.size)
    fg_to_bg = bg_to_fg = wrong_fg = correct_auto = deferred = 0
    correct_fg = correct_bg = fg_resolve_total = bg_clear_total = 0
    n_fg = int(np.sum(y[selected] != 0))
    n_bg = n - n_fg

    for held_pos, held_idx in enumerate(selected.tolist()):
        keep = np.arange(n, dtype=np.int64) != int(held_pos)
        support_idx = selected[keep]
        support_y = y[support_idx]
        thresholds = class_support_thresholds(fg_x[support_idx], support_y, support_quantile)
        bio_labels = _topk_labels(fg_x[held_idx], fg_x[support_idx], support_y, k)
        dino_labels = _topk_labels(bg_x[held_idx], bg_x[support_idx], support_y, k)

        fg_counts = Counter(int(v) for v in bio_labels.tolist() if int(v) != 0)
        if fg_counts:
            max_fg_vote = max(int(v) for v in fg_counts.values())
            winners = sorted(int(cls) for cls, votes in fg_counts.items() if int(votes) == max_fg_vote)
            winning_cls = int(winners[0])
            unique = len(winners) == 1
        else:
            max_fg_vote = 0
            winning_cls = 0
            unique = False
        dino_bg_votes = int(np.sum(dino_labels == 0))
        cls_cols = np.flatnonzero(support_y == winning_cls) if winning_cls != 0 else np.asarray([], dtype=np.int64)
        floor = thresholds.get(int(winning_cls), float("inf"))
        sim = float(np.max(fg_x[held_idx] @ fg_x[support_idx[cls_cols]].T)) if cls_cols.size else float("nan")
        floor_ok = bool(np.isfinite(floor) and sim >= float(floor))

        clear = dino_bg_votes >= int(tau_bg) and not (max_fg_vote >= int(tau_veto))
        fg_condition = (
            max_fg_vote >= int(tau_fg) and unique and int(cls_cols.size) >= 2 and floor_ok
        )
        resolve = fg_condition if foreground_first else ((not clear) and fg_condition)
        true_label = int(y[held_idx])

        if foreground_first and resolve:
            fg_resolve_total += 1
            if true_label != 0 and winning_cls == true_label:
                correct_auto += 1
                correct_fg += 1
            elif true_label == 0:
                bg_to_fg += 1
            else:
                wrong_fg += 1
        elif clear:
            bg_clear_total += 1
            if true_label == 0:
                correct_auto += 1
                correct_bg += 1
            else:
                fg_to_bg += 1
        elif resolve:
            fg_resolve_total += 1
            if true_label != 0 and winning_cls == true_label:
                correct_auto += 1
                correct_fg += 1
            elif true_label == 0:
                bg_to_fg += 1
            else:
                wrong_fg += 1
        else:
            deferred += 1

    return {
        "samples": n,
        "foreground_samples": n_fg,
        "background_samples": n_bg,
        "k": int(k),
        "tau_fg": int(tau_fg),
        "tau_bg": int(tau_bg),
        "tau_veto": int(tau_veto),
        "fg_to_bg_count": int(fg_to_bg),
        "fg_to_bg_rate": float(fg_to_bg / n_fg) if n_fg else 0.0,
        "correct_auto_count": int(correct_auto),
        "correct_auto_pct": 100.0 * float(correct_auto / n) if n else 0.0,
        "deferred_count": int(deferred),
        "deferred_pct": 100.0 * float(deferred / n) if n else 0.0,
        "bg_to_fg_count": int(bg_to_fg),
        "wrong_fg_class_count": int(wrong_fg),
        "fg_resolve_precision": float(correct_fg / fg_resolve_total) if fg_resolve_total else float("nan"),
        "bg_clear_precision": float(correct_bg / bg_clear_total) if bg_clear_total else float("nan"),
        "fg_resolve_count": int(fg_resolve_total),
        "bg_clear_count": int(bg_clear_total),
    }


def calibrate_rule(
    bg_x: np.ndarray,
    fg_x: np.ndarray,
    y: np.ndarray,
    selected: Sequence[int],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    support_quantile: float = 0.5,
    calibration_mode: str = "free_thresholds",
    foreground_first: bool = False,
) -> Tuple[TriageRule, Dict[str, float | int], List[Dict[str, float | int]]]:
    """Select the guarded triage rule by LOO over the calibration bank.

    Returns ``(rule, winning_row, all_rows)`` where ``rule`` is the chosen
    :class:`TriageRule`, ``winning_row`` its LOO diagnostics, and ``all_rows`` every
    candidate's LOO diagnostics (for auditing the calibration).
    """
    selected_arr = np.asarray(selected, dtype=np.int64)
    rows: List[Dict[str, float | int]] = []
    for k in k_values:
        for tau_fg, tau_bg, tau_veto in candidate_thresholds(int(k), calibration_mode):
            rows.append(
                loo_score(
                    bg_x, fg_x, y, selected_arr,
                    int(k), int(tau_fg), int(tau_bg), int(tau_veto),
                    float(support_quantile), foreground_first,
                )
            )
    if not rows:
        raise ValueError("No candidate thresholds evaluated (empty k_values?).")
    ranked = sorted(
        rows,
        key=lambda r: (
            int(r["fg_to_bg_count"]),
            float(r["fg_to_bg_rate"]),
            -float(r["correct_auto_pct"]),
            float(r["deferred_pct"]),
            int(r["k"]),
            -int(r["tau_fg"]),
            -int(r["tau_bg"]),
            -int(r["tau_veto"]),
        ),
    )
    best = ranked[0]
    rule = TriageRule(
        k=int(best["k"]),
        tau_fg=int(best["tau_fg"]),
        tau_bg=int(best["tau_bg"]),
        tau_veto=int(best["tau_veto"]),
        support_quantile=float(support_quantile),
        foreground_first=bool(foreground_first),
    )
    return rule, best, rows
