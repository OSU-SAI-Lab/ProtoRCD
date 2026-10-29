# Ground-truth decoupling: research code → production pipeline

This repo ports the stage-3 acquisition + triage logic from the WACV research code and
removes its dependency on ground-truth (GT) boxes at the **decision** level, so a UI can
drive review on live SAM3 proposals. This note records exactly what was ported, from
where, and where GT was cut — so the port can be audited against the paper.

## Source modules

| Production module | Ported from (research `scripts/`) | Functions |
|---|---|---|
| `triage/acquisition.py` | `evaluate_active_triage_features.py` | `acquire`, `select_weighted_kcenter`, `select_kcenter`, `compute_density`, `full_knn_percentile_density`, `approximate_density`, `density_percentile_ranks`, `l2_normalize` |
| `triage/triage.py` | `evaluate_two_stage_calibrated_guarded_initial100.py` | `guarded_triage_with_diagnostics` → `guarded_triage`; `class_support_thresholds`, `topk_indices_sims` |
| `triage/calibration.py` | `evaluate_two_stage_calibrated_guarded_initial100.py` | `loo_score`, `calibrate_rule`, `candidate_thresholds` |
| `triage/pool.py` | `evaluate_active_triage_features.py` | `load_feature_store`, `label_metadata` → `labels_from_coco`, `pairwise_iou_one_to_many`, `xywh_to_xyxy` |
| `triage/session.py` | `evaluate_final_shared_random_calibrated_zero_start_protocol_ablation.py` (`build_initial_state` + round loop) | new orchestration |

The decision logic is ported **logic-for-logic** (same vote counting, same veto handling,
same per-class similarity floor, same LOO ranking) so that, given the same labels, the
production rule reproduces the research rule.

> **This note is about GT decoupling, not about which configuration is canonical.** The
> ported modules are *general* (see "Two configurations" below): they can run a BioCLIP-2
> veto and a free-threshold calibration grid. The validated production protocol is
> `split_no_veto`. Do not read the guarded-veto description here as the production default.

## Where ground truth entered the research code

Two places only:

1. **Label assignment.** `label_metadata()` matched each proposal to COCO GT boxes by
   IoU (`pairwise_iou_one_to_many`, threshold 0.25) and assigned the best-matching box's
   category (else background = 0). This is "uses GT boxes."
2. **Diagnostics.** `guarded_triage_with_diagnostics` read `true_y[qidx]` to log
   `gt_label` / `gt_is_foreground` and to score errors. The decision itself never used
   `true_y`.

Crucially, **the acquisition policy and the triage decision were already GT-free** — they
operate only on embeddings and on the *reviewed support set*. GT only ever supplied the
labels of reviewed proposals and the numbers for evaluation.

## What changed in the port

- **Label source is now a `LabelProvider`** (`triage/labels.py`). The reviewed label of a
  proposal comes from:
  - `CallbackLabelProvider` — a UI callback `request(ids) -> {id: label}` (primary).
  - `FileLabelProvider` — a per-round JSON exchange for out-of-process UIs (built on the
    callback contract).
  - `GTLabelProvider` — the old IoU-to-COCO behavior, kept for offline reproducibility
    and tests. `labels_from_coco()` is the ported `label_metadata`.
- **`true_y` is optional in `guarded_triage`.** It is accepted only to attach `gt_*`
  diagnostic fields when labels happen to be known (offline/eval). It is never read in
  any branch of the decision. `tests/test_triage_session.py::test_triage_decision_is_gt_independent`
  asserts the decisions are byte-identical with and without `true_y`.
- **The session owns the partition.** `TriageSession` keeps three disjoint sets over the
  pool: the calibration bank (drawn once from `seed`, held out forever), the reviewed
  support set (grows each round), and the available pool. This matches the research
  protocol: calibration proposals are excluded from acquisition, from the support set,
  and from the triage query pool.

## Invariants preserved from the paper

- **Two representations, asymmetric roles.** Background/Clear evidence = DINOv3 (`bg_x`);
  foreground/Resolve evidence = BioCLIP-2 (`fg_x`).
- **Canonical acquisition** = `density_balanced_kcenter` ("density-moderated"), weighting
  farthest-point selection by `q·(1−q)` on the density percentile.
- **Integer vote thresholds** chosen by leave-one-out over the calibration bank, ranked
  safety-first (min FG→BG count, min FG→BG rate, max correct-auto %, min defer %, then
  conservative tie-breaks). Thresholds are counts, not probabilities.
- **Per-seed calibration bank** reproduced by `default_rng(seed).choice(N, 100)`, held out
  from acquisition, from the support set, and from the triage query pool.
- **Only human-reviewed labels become prototypes;** auto Resolve/Clear decisions are
  retired and never reused as supervision.

## Two configurations: generic capability vs the frozen protocol

The ported modules are deliberately general. Do not confuse the **generic capability** of
`triage.py` / `calibration.py` with the **validated production protocol**.

**Generic ported implementation** (retained for compatibility / experimentation):

- supports a BioCLIP-2 **veto** on the Clear rule (`tau_veto`);
- supports a **free-threshold calibration grid** over `k × (tau_fg, tau_bg, tau_veto)`;
- reachable via the plain `TriageSession(...)` constructor with explicit knobs.

**Frozen canonical research protocol — `split_no_veto`** (what production and every
reproducible result use; instantiate with `TriageSession.for_protocol("split_no_veto",
...)`):

- BioCLIP-2 = foreground / species evidence; DINOv3 = background evidence;
- **Resolve evaluated before Clear** (`foreground_first`) — the burden to *drop* an animal
  is higher than to *keep* it;
- **veto disabled** (`tau_veto = k + 1`);
- **semantic-supermajority** calibration (`tau_fg = tau_bg = ⌊2k/3⌋ + 1`), one candidate
  per `k` rather than the free grid;
- **k ∈ {5, 7, 9, 11}**;
- class-specific similarity floor uses the canonical **`support_quantile = 0.0`**;
- **initial-scope** density (computed once over the kept pool, then sliced per round);
- human-reviewed examples only enter the prototype bank; auto Resolve/Clear are retired.

The earlier "guarded veto" wording in the research code describes the *generic capability*.
The frozen protocol disables the veto — see
[`../protorcd_pipeline/triage/README.md`](../protorcd_pipeline/triage/README.md) for the full
decision rule and [`../regression/README.md`](../regression/README.md) for the bit-for-bit
validation.

## What was intentionally left out

The research suite's experiment drivers, SLURM launchers, aggregation/plotting scripts,
the frozen results dossier, and all the alternative baselines (OSLO, Tip-Adapter,
Symmetric-kNN, linear/LR probes, pseudo-label variants) are **not** in this repo. They
are evaluation machinery, not part of the deployable detect→features→triage pipeline.
