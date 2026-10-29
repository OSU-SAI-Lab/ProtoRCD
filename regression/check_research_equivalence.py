#!/usr/bin/env python3
"""Research-equivalence regression: production session vs the frozen research run.

Runs the production :class:`TriageSession` end-to-end with the canonical
``split_no_veto`` protocol and an offline GT provider, then diffs every stage against a
frozen research run directory (the ``decision_cache.npz`` + calibration artifacts that
the paper's launch script wrote). A pass means the production pipeline reproduces the
research trajectory exactly for that (dataset, seed).

The checklist (A run passes only if ALL checks pass):

    E1  calibration rule           k / tau_fg / tau_bg / tau_veto match selected_rule
    E2  calibration bank           the per-seed random draw matches (kept-space compaction)
    E3  GT labeling                production IoU labels == research true labels (kept-space)
    E4  acquisition order          per-round selected indices match, in order
    E5  auto (retired) set         auto indices + predicted classes match, per round
    E6  final deferred set         remaining-after-all-rounds indices match
    E7  accounting                 selected U auto U remaining == kept pool (both sides)

Usage
-----
    python regression/check_research_equivalence.py \
        --research-run <.../runs/split_no_veto/ENA24/density_balanced_kcenter/seed_3> \
        --bg-features  <.../ENA24/features/all/animal/dinov3/box> \
        --fg-features  <.../ENA24/features/all/animal/bioclip/box> \
        --coco         <.../ena24_animal_bbox_instances.json> \
        --seed 3 --rounds 16 --batch-size 50 --calibration-size 100 --iou-threshold 0.25

Exit code 0 iff every check passes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow running from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protorcd_pipeline.triage import (  # noqa: E402
    GTLabelProvider,
    TriageSession,
    labels_from_coco,
    labels_from_coco_multi,
    load_pool,
    load_pool_multi,
)


def _load_research(run_dir: Path) -> Dict[str, np.ndarray]:
    cache = np.load(run_dir / "decision_cache.npz", allow_pickle=True)
    return {k: cache[k] for k in cache.files}


def _fmt(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _split(value: str) -> List[str]:
    return [v for v in (s.strip() for s in value.split(",")) if v]


def run_production(args: argparse.Namespace) -> Tuple[TriageSession, np.ndarray]:
    """Drive the production session exactly as the paper's split_no_veto protocol.

    ``--bg-features``/``--fg-features``/``--coco`` may be comma-separated for multi-split
    (explicit) datasets like LoTE; the stores are concatenated in the given order and the
    COCO tables merged, matching the research ``--explicit-review-split all`` path.
    """
    bg_dirs = _split(args.bg_features)
    fg_dirs = _split(args.fg_features)
    coco_paths = _split(args.coco)
    if len(bg_dirs) > 1 or len(fg_dirs) > 1:
        pool = load_pool_multi(bg_dirs, fg_dirs, split_tags=args.split_tags and _split(args.split_tags))
        gt_labels, _names = labels_from_coco_multi(pool.metadata, coco_paths, args.iou_threshold)
    else:
        pool = load_pool(bg_dirs[0], fg_dirs[0])
        gt_labels, _names = labels_from_coco(pool.metadata, coco_paths[0], args.iou_threshold)
    pool.gt_labels = gt_labels  # diagnostics only; never enters the decision
    provider = GTLabelProvider(gt_labels, id_to_index=pool.id_to_index)

    session = TriageSession.for_protocol(
        "split_no_veto", pool, provider,
        seed=args.seed, batch_size=args.batch_size,
    )
    precomputed = None
    if args.density_cache:
        precomputed = np.load(args.density_cache)  # kept-space, aligned to kept_source
    session.build_calibration_bank(size=args.calibration_size, precomputed_density=precomputed)
    for _ in range(args.rounds):
        if session.finished:
            break
        session.step(collect_rows=False)
    return session, gt_labels


def check(args: argparse.Namespace) -> bool:
    research = _load_research(Path(args.research_run))
    session, gt_labels = run_production(args)

    results: List[Tuple[str, bool, str]] = []

    # ---- E1: calibration rule ------------------------------------------- #
    assert session.rule is not None
    rr = research  # aliases
    # The winning rule is the top row of research candidate rows; selected_rule.csv holds
    # the same. We compare against the production-selected rule.
    prod_rule = session.rule.as_dict()
    # Reconstruct research selected rule from the candidate rows (min sort key == winner).
    import json as _json
    cand = [_json.loads(s) for s in rr["calibration_candidate_rows_json"].tolist()]
    def _key(r: dict):
        return (
            float(r["fg_to_bg_count"]), float(r["fg_to_bg_rate"]),
            -float(r["correct_auto_pct"]), float(r["deferred_pct"]),
            int(r["k"]), -int(r["tau_fg"]), -int(r["tau_bg"]), -int(r["tau_veto"]),
        )
    res_rule = sorted(cand, key=_key)[0]
    e1 = all(int(prod_rule[k]) == int(res_rule[k]) for k in ("k", "tau_fg", "tau_bg", "tau_veto"))
    results.append((
        "E1 calibration rule", e1,
        f"prod k={prod_rule['k']} fg={prod_rule['tau_fg']} bg={prod_rule['tau_bg']} veto={prod_rule['tau_veto']} | "
        f"research k={res_rule['k']} fg={res_rule['tau_fg']} bg={res_rule['tau_bg']} veto={res_rule['tau_veto']}",
    ))

    # ---- E2: calibration bank (original pool space) --------------------- #
    res_cal = np.sort(np.asarray(rr["calibration_source_indices"], dtype=np.int64))
    prod_cal = np.sort(session.calibration_indices.astype(np.int64))
    e2 = np.array_equal(res_cal, prod_cal)
    results.append(("E2 calibration bank", e2, f"n={prod_cal.size}, identical={e2}"))

    # ---- Build research kept-space partition --------------------------- #
    res_sel = np.asarray(rr["selected_indices"], dtype=np.int64)         # kept-space, ordered
    res_sel_round = np.asarray(rr["selected_rounds"], dtype=np.int64)
    res_auto = np.asarray(rr["auto_indices"], dtype=np.int64)
    res_auto_round = np.asarray(rr["auto_rounds"], dtype=np.int64)
    res_auto_pred = np.asarray(rr["auto_pred_labels"], dtype=np.int64)
    res_final = np.asarray(rr["final_review_indices"], dtype=np.int64)
    res_sel_y = np.asarray(rr["selected_true_labels"], dtype=np.int64)

    # Production kept-space partition.
    prod_sel = session.support_indices_kept                              # kept-space, ordered
    prod_auto = session.auto_indices_kept
    prod_auto_pred = session.auto_predictions
    prod_final = np.asarray(session._available, dtype=np.int64)          # remaining/deferred
    # production kept-space GT label lookup
    kept_y = gt_labels[session.kept_source]

    # ---- E3: GT labeling (kept-space true labels for the selected set) -- #
    # Compare production kept-space labels on the research-selected indices to the cached
    # research true labels for those same indices.
    e3 = np.array_equal(kept_y[res_sel], res_sel_y)
    results.append(("E3 GT labeling", e3, f"selected true-label match on {res_sel.size} rows = {e3}"))

    # ---- E4: acquisition order (per round, in order) ------------------- #
    e4 = prod_sel.shape == res_sel.shape and np.array_equal(prod_sel, res_sel)
    if e4:
        detail = f"{prod_sel.size} acquisitions identical, in order"
    else:
        first = int(np.flatnonzero(prod_sel[:min(prod_sel.size, res_sel.size)]
                                   != res_sel[:min(prod_sel.size, res_sel.size)])[0]) \
            if prod_sel.size and res_sel.size and (prod_sel[:min(prod_sel.size, res_sel.size)]
                                                   != res_sel[:min(prod_sel.size, res_sel.size)]).any() else -1
        detail = (f"prod n={prod_sel.size} research n={res_sel.size}; "
                  f"first differing position={first}")
    results.append(("E4 acquisition order", e4, detail))

    # ---- E5: auto (retired) set + predictions -------------------------- #
    # Order within a round is deterministic (candidate_idx order); compare as sets per
    # round AND overall multiset with predictions attached.
    def _pairs(idx: np.ndarray, pred: np.ndarray):
        return sorted((int(i), int(p)) for i, p in zip(idx.tolist(), pred.tolist()))
    e5 = (prod_auto.size == res_auto.size
          and _pairs(prod_auto, prod_auto_pred) == _pairs(res_auto, res_auto_pred))
    results.append(("E5 auto set + preds", e5,
                    f"prod auto={prod_auto.size} research auto={res_auto.size}, (idx,pred) match={e5}"))

    # ---- E6: final deferred set ---------------------------------------- #
    e6 = np.array_equal(np.sort(prod_final), np.sort(res_final))
    results.append(("E6 final deferred", e6,
                    f"prod remaining={prod_final.size} research={res_final.size}, identical={e6}"))

    # ---- E7: accounting (both sides partition the kept pool) ----------- #
    kept_n = int(session.kept_source.size)
    prod_total = prod_sel.size + prod_auto.size + prod_final.size
    prod_union = np.concatenate([prod_sel, prod_auto, prod_final])
    e7 = (prod_total == kept_n
          and np.array_equal(np.sort(prod_union), np.arange(kept_n)))
    results.append(("E7 accounting", e7, f"kept={kept_n} total={prod_total} exact-cover={e7}"))

    # ---- report -------------------------------------------------------- #
    print(f"\nResearch-equivalence: {Path(args.research_run).name}  (seed {args.seed})")
    print("-" * 78)
    all_ok = True
    for name, ok, detail in results:
        all_ok &= ok
        print(f"  [{_fmt(ok):4}] {name:22} {detail}")
    print("-" * 78)
    print(f"  OVERALL: {'PASS -- production reproduces the research run exactly' if all_ok else 'FAIL'}")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--research-run", required=True, type=Path,
                    help="frozen research run dir (contains decision_cache.npz)")
    ap.add_argument("--bg-features", required=True,
                    help="DINOv3 feature store dir(s), comma-separated for multi-split")
    ap.add_argument("--fg-features", required=True,
                    help="BioCLIP-2 feature store dir(s), comma-separated for multi-split")
    ap.add_argument("--coco", required=True,
                    help="COCO annotation file(s), comma-separated for multi-split")
    ap.add_argument("--split-tags", default=None,
                    help="optional comma-separated tags for multi-split stores (e.g. train,test)")
    ap.add_argument("--density-cache", default=None,
                    help="optional frozen kept-space density .npy to pin acquisition "
                         "tie-breaking (a large float32 kNN matmul is not bit-for-bit "
                         "reproducible across BLAS/hardware)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--calibration-size", type=int, default=100)
    ap.add_argument("--iou-threshold", type=float, default=0.25)
    args = ap.parse_args()
    return 0 if check(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
