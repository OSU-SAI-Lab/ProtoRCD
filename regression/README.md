# Research-equivalence regression

Proves the production triage pipeline reproduces the **frozen research runs** — the runs
the paper's numbers come from — bit-for-bit, per (dataset, seed).

**Baselines this guards** (see the [root README](../README.md) for all provenance tags):

| Tag | Commit | Role |
|---|---|---|
| `scientific-parity-baseline` | `08e0fa0` | the frozen scientific algorithms this harness validates |
| `serving-hardened-baseline` | `21fb490` | current handoff; `protorcd_pipeline/` byte-identical to `08e0fa0`, so parity holds unchanged |

**Verified result at the hardened baseline:**

- **70 / 70** equivalence checks PASS (10 runs × 7 checks E1–E7)
- **10 / 10** (dataset, seed) runs OVERALL PASS — ENA24 + LoTE, seeds 3/7/11/13/17

## What it checks

[`check_research_equivalence.py`](check_research_equivalence.py) drives the production
`TriageSession` with the canonical `split_no_veto` protocol and an offline `GTLabelProvider`,
then diffs every stage against a research run's `decision_cache.npz`:

| # | Check | What must match |
|---|---|---|
| E1 | calibration rule | `k, tau_fg, tau_bg, tau_veto` |
| E2 | calibration bank | the per-seed random draw (original pool space) |
| E3 | GT labeling | production IoU-to-COCO labels == research true labels (kept-space) |
| E4 | acquisition order | per-round selected indices, **in order** |
| E5 | auto (retired) set | auto indices **and** predicted classes |
| E6 | final deferred set | remaining-after-all-rounds indices |
| E7 | accounting | selected ⊎ auto ⊎ remaining == kept pool |

A run passes only if **all** checks pass. Exit code is 0 iff it passed.

## Running

```bash
# one (dataset, seed):
python regression/check_research_equivalence.py \
    --research-run <.../runs/split_no_veto/ENA24/density_balanced_kcenter/seed_3> \
    --bg-features  <.../ENA24/features/all/animal/dinov3/box> \
    --fg-features  <.../ENA24/features/all/animal/bioclip/box> \
    --coco         <.../ena24_animal_bbox_instances.json> \
    --seed 3

# the whole matrix (ENA24 + LoTE, seeds 3,7,11,13,17):
regression/run_all.sh
```

Multi-split datasets (e.g. LoTE = train + test pooled) pass comma-separated
`--bg-features/--fg-features/--coco` and `--split-tags train,test`; the harness
concatenates the stores in order and merges the COCO tables, matching the research
`--explicit-review-split all` path.

## The density caveat (why `--density-cache` exists)

Every piece of the triage *logic* is deterministic and reproduces exactly. The one
non-deterministic input is the **local density**: it is a large float32 kNN matmul, and
float32 matmul reduction order is not identical across BLAS builds / hardware. The
research density cache was written on one such environment; recomputing it elsewhere
yields values that differ by ~1e-4.

On small pools (ENA24, ~11k proposals) those differences never change a farthest-point
selection, so recomputation is bit-identical — all five ENA24 seeds pass without any
cache. On large, dense pools (LoTE, ~40k) a ~1e-4 density difference can flip a handful of
near-ties in the k-center greedy step, which then cascades. To reproduce such a run
exactly, pass the frozen research density with `--density-cache <...npy>`; the session
consumes it via `build_calibration_bank(precomputed_density=...)` instead of recomputing.
`run_all.sh` does this automatically for LoTE.

This is not a production defect — the research code is not self-reproducible on those ties
either. It is a property of float32 linear algebra. For a live deployment the density is
computed once per session and is perfectly deterministic *within* that environment.
