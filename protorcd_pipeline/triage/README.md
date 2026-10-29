# `protorcd_pipeline.triage` — active acquisition + Resolve/Clear/Defer

Stage 3 of the pipeline. Given two frozen representations of each detector proposal —
**BioCLIP-2** (foreground / species) and **DINOv3** (background / validity) — this
package runs the paper's active-review loop: pick a small batch to send for human
review, then use the reviewed set as prototypes to auto-decide the rest.

> **Generic module vs frozen protocol.** [`triage.py`](triage.py) and
> [`calibration.py`](calibration.py) are *general* implementations — they can run with a
> BioCLIP-2 veto and a free-threshold calibration grid, kept for experimentation. The
> **validated production configuration is `split_no_veto`** (veto disabled,
> semantic-supermajority calibration, Resolve-before-Clear). Everything below describes
> `split_no_veto`; always instantiate it via `TriageSession.for_protocol("split_no_veto",
> ...)`, never by hand-setting thresholds.

It is **decoupled from ground truth**: acquisition and the triage decision never read GT
boxes. Labels arrive only through a [`LabelProvider`](labels.py) (a UI callback, a JSON
file exchange, or — offline only — an IoU-to-COCO oracle). That is the seam that lets the
same code drive a live UI and reproduce the paper's numbers.

## The two inference problems

Ecological detector-proposal triage is really two questions per proposal:

1. **Species identity** — *what animal is this?* Answered by BioCLIP-2 kNN votes (`fg_x`).
2. **Proposal validity** — *is this a real animal box at all, or background?* Answered by
   DINOv3 kNN "is-background" votes (`bg_x`).

The rule keeps them separate and asymmetric, which is the paper's central claim.

## The three decisions

For each unreviewed proposal, given the reviewed support set (embeddings + human labels,
background == class 0):

| Decision | Meaning | Condition (canonical `split_no_veto`) |
|---|---|---|
| **Resolve** | auto-label with a species | winning FG species has ≥ `tau_fg` votes, is the unique top class, has ≥ 2 reviewed exemplars, and clears its per-class similarity floor |
| **Clear** | auto-mark as background | DINOv3 gives ≥ `tau_bg` background votes (the veto is disabled in `split_no_veto`) |
| **Defer** | send to a human | everything else |

In `split_no_veto`, **Resolve is evaluated strictly before Clear** (`foreground_first`):
a proposal eligible for both is Resolved, never Cleared. This is what makes the rule
asymmetric — the burden of proof to *drop* an animal is higher than to *keep* it.

## Files

| File | Role |
|---|---|
| [`acquisition.py`](acquisition.py) | density estimates + selection policies (`density_balanced_kcenter` = the paper's "density-moderated" k-center). Pure geometry, GT-free. |
| [`triage.py`](triage.py) | the Resolve/Clear/Defer decision (`guarded_triage`) + per-class similarity floors + the `TriageRule` dataclass. Supports the veto generically; `split_no_veto` disables it. |
| [`calibration.py`](calibration.py) | leave-one-out calibration of `(k, tau_fg, tau_bg, tau_veto)`. Supports both the free-threshold grid (generic) and the semantic-supermajority mode (`split_no_veto`). |
| [`labels.py`](labels.py) | the `LabelProvider` seam: `CallbackLabelProvider` (UI), `FileLabelProvider` (out-of-process UI), `GTLabelProvider` (offline oracle). |
| [`pool.py`](pool.py) | load two-representation proposal pools from stage-2 feature stores; offline COCO IoU labeling. |
| [`session.py`](session.py) | the round-loop orchestration a UI drives: `TriageSession`. |

## The round loop (what a UI drives)

```python
from protorcd_pipeline.triage import TriageSession, CallbackLabelProvider, load_pool

pool = load_pool(dinov3_store, bioclip_store)              # bg_x, fg_x, ids, metadata
provider = CallbackLabelProvider(show_crops_and_get_labels) # your UI

# Canonical paper protocol — sets foreground_first, veto off, supermajority
# calibration, k in {5,7,9,11}, density-moderated k-center, initial-scope density.
session = TriageSession.for_protocol("split_no_veto", pool, provider, seed=3)

session.build_calibration_bank(size=100)   # draw + review + calibrate the rule
while not session.finished:
    info = session.step()                  # acquire 50 -> label -> triage -> RETIRE autos
    # info.triage : this round's Resolve/Clear/Defer over the remaining pool
    # info.retired: how many proposals were auto-decided and left the pool this round
```

`step()` is the **complete round**: it acquires the batch, gets its labels, triages the
remaining pool with the updated bank, and retires the auto-decisions — all in one call.
Do not call a separate triage step after it.

The plain `TriageSession(...)` constructor exists for experimentation with the generic
capabilities (custom thresholds, the veto, a free calibration grid). Production and every
reproducible result use `for_protocol("split_no_veto")`; see [`session.py`](session.py)
for the individual knobs.

## The state machine (and the invariant that guards it)

The **kept pool** = every proposal except the calibration bank. Each round partitions it
into three disjoint sets whose union is always the whole kept pool:

```
reviewed (support)   ⊎   auto-decided (retired)   ⊎   remaining (deferred)   ==   kept pool
```

* **reviewed** — acquired for human review. *Only these become prototypes.*
* **auto-decided** — Resolved or Cleared. They **leave the pool permanently** and are
  never re-acquired. Their predictions are **not** used as supervision.
* **remaining** — deferred; the only proposals eligible for the next round.

`TriageSession.step()` enforces this partition every round (`_assert_round_accounting`).
The single most important correctness property of the loop is that **auto-decisions are
retired** — without that, a Cleared/Resolved proposal could be acquired again and pollute
the support set. (This mirrors the research reference's
`review_idx = candidate_idx[~auto]` + `assert_round_accounting`.)

## Kept-space vs pool-space indices

`build_calibration_bank()` removes the calibration rows and **compacts** the remaining
pool to `arange(N - calibration_size)` — "kept-space". This matches the research decision
cache, whose indices are kept-space. Two views are exposed:

* pool-space (`calibration_indices`, `support_indices`) — indices into the original pool.
* kept-space (`support_indices_kept`, `auto_indices_kept`, `kept_source`) — indices into
  the compacted pool; `kept_source[k]` maps kept row `k` back to its original pool row.

Proposal **ids** (from the `LabelProvider`) are always resolved through pool-space, so a
UI never has to know about kept-space.

## Reproducing the paper exactly

The offline path (`GTLabelProvider` + `for_protocol("split_no_veto")`) reproduces the
research trajectory **bit-for-bit** — same calibration rule, same acquisition order, same
auto-decisions, same deferrals — across all seeds. See
[`../../regression/`](../../regression/) for the equivalence harness and how it is
verified against the frozen research runs.

One caveat, documented there: a large float32 kNN **density** matmul is not bit-for-bit
reproducible across BLAS/hardware. On big, dense pools that can flip a handful of
farthest-point *ties*. For exact reproduction on such pools, pass the frozen density to
`build_calibration_bank(..., precomputed_density=...)`; on smaller pools recomputation is
already identical.
