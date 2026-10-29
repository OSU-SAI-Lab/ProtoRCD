#!/usr/bin/env python3
"""Offline end-to-end triage run (reproducibility mode -- no UI).

Given two feature stores (DINOv3 = background, BioCLIP-2 = foreground) extracted from
the SAME SAM3 proposals, plus the dataset's COCO annotations, this drives the full
active-triage loop using a GTLabelProvider (labels come from IoU-matching to GT boxes).

This is the sanity/reproducibility path. For a real deployment, swap GTLabelProvider for
CallbackLabelProvider (in-process UI) or FileLabelProvider (out-of-process UI); the loop
code below is identical.

Example:
    python examples/run_triage_offline.py \
        --bg-features /path/to/dinov3_store \
        --fg-features /path/to/bioclip_store \
        --coco /path/to/annotations.json \
        --batch-size 50 --rounds 16 --seed 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from protorcd_pipeline.triage import (
    GTLabelProvider,
    TriageSession,
    labels_from_coco,
    load_pool,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bg-features", required=True, help="DINOv3 feature store dir")
    ap.add_argument("--fg-features", required=True, help="BioCLIP-2 feature store dir")
    ap.add_argument("--coco", required=True, help="COCO annotations for GT-mode labels")
    ap.add_argument("--iou-threshold", type=float, default=0.25)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--calibration-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    # 1. Load the two-representation proposal pool from stage-2 feature stores.
    pool = load_pool(args.bg_features, args.fg_features)
    print(f"Loaded pool: {pool.size} proposals "
          f"(bg dim {pool.bg_x.shape[1]}, fg dim {pool.fg_x.shape[1]})")

    # 2. OFFLINE ONLY: derive GT labels by IoU-matching proposals to COCO boxes.
    gt_labels, names = labels_from_coco(pool.metadata, args.coco, args.iou_threshold)
    pool.gt_labels = gt_labels  # attaches diagnostics; NOT used by the decision
    n_fg = int(np.sum(gt_labels != 0))
    print(f"GT labels: {n_fg} foreground / {pool.size - n_fg} background "
          f"across {len(names) - 1} species")

    # ids in the feature store are stable proposal_ids (e.g. "1014:0"), not positions,
    # so key the offline oracle by id_to_index.
    provider = GTLabelProvider(gt_labels, id_to_index=pool.id_to_index)

    # 3. Run the loop -- identical to the UI path, only the provider differs.
    #    for_protocol("split_no_veto") applies the validated paper configuration
    #    (foreground_first, veto disabled, semantic-supermajority calibration,
    #    k in {5,7,9,11}, density-moderated k-center, initial-scope density).
    session = TriageSession.for_protocol(
        "split_no_veto", pool, provider,
        seed=args.seed, batch_size=args.batch_size,
    )
    session.build_calibration_bank(size=args.calibration_size)
    print(f"Calibrated rule: {session.rule.as_dict()}")

    # One step() is a complete round: acquire -> label -> triage -> retire auto-decisions.
    # info.triage holds this round's Resolve/Clear/Defer over the remaining pool.
    for _ in range(args.rounds):
        if session.finished:
            break
        info = session.step(collect_rows=False)
        result = info.triage
        c = result.counts()
        total = max(1, len(result.decisions))
        auto = c["resolve"] + c["clear"]
        print(f"round {info.round:2d} | reviewed {info.cumulative_reviews:5d} | "
              f"remaining {info.remaining:6d} | "
              f"resolve {c['resolve']:5d} clear {c['clear']:5d} defer {c['defer']:5d} | "
              f"auto-coverage {100.0 * auto / total:5.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
