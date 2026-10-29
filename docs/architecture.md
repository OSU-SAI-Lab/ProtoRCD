# Architecture — how the pipeline fits together

Start here if you are new to the repo. This is the conceptual map; the deep material lives
in the sub-READMEs this document links to. It does **not** repeat the HTTP schema
([`serving/API.md`](../serving/API.md)) or the triage equations
([`protorcd_pipeline/triage/README.md`](../protorcd_pipeline/triage/README.md)).

## The pipeline, end to end

```
imagery
  -> SAM3 proposals                    (class-agnostic animal boxes)
  -> DINOv3 + BioCLIP-2 feature stores (two aligned embeddings per proposal)
  -> TriageSession                     (active review loop)
  -> Resolve / Clear / Defer           (a decision per proposal)
```

Every proposal ends up in exactly one of three states:

- **Resolve** — auto-labelled with a species (foreground).
- **Clear** — auto-marked as background / not-an-animal.
- **Defer** — sent to a human.

The two feature stores are the hinge. **DINOv3** answers *"is this a real animal box at
all?"* (proposal validity / background evidence); **BioCLIP-2** answers *"which species is
it?"* (foreground evidence). Keeping those two questions on two representations is the
scientific core of the method.

## Two ways to run it

`TriageSession` is the single entry point. It runs the same way regardless of who supplies
the human labels:

```
              TriageSession   (the frozen scientific loop)
                 /        \
   headless Python/scripts   serving/ HTTP API  ->  external UI
   (labels from an offline    (labels from a browser client,
    GT oracle, or a callback)   one HTTP round-trip per batch)
```

- **Headless** — a script drives the loop directly and supplies labels through a
  `LabelProvider` (an offline GT oracle for reproducibility, or an in-process callback).
  See [`examples/`](../examples/README.md). *The pipeline must always remain runnable this
  way — the UI is optional.*
- **HTTP / UI** — [`serving/`](../serving/README.md) wraps one `TriageSession` per session
  behind a small HTTP API so an external UI can review batches. This is the recommended
  path for a browser front-end.

## Directory responsibilities

| Directory | Responsibility | Frozen? |
|---|---|---|
| [`protorcd_pipeline/detect/`](../protorcd_pipeline/detect/) | Stage 1 — SAM3 text-prompt detection → class-agnostic proposals | scientific |
| [`protorcd_pipeline/features/`](../protorcd_pipeline/features/) | Stage 2 — DINOv3 + BioCLIP-2 embeddings per proposal | scientific |
| [`protorcd_pipeline/common/`](../protorcd_pipeline/common/) | Detect→features bridge; shared records / geometry / IO | scientific |
| [`protorcd_pipeline/triage/`](../protorcd_pipeline/triage/README.md) | Stage 3 — active acquisition + Resolve/Clear/Defer | **scientifically frozen** |
| [`serving/`](../serving/README.md) | Hardened HTTP adapter exposing `TriageSession` to a UI | UI-facing adapter |
| [`examples/`](../examples/README.md) | Runnable headless + provider examples | examples |
| `tests/` | Fast unit/integration tests (triage + serving) | tests |
| [`regression/`](../regression/README.md) | Bit-for-bit research-equivalence harness | verification |

### The rules that keep the layers honest

- **`protorcd_pipeline/triage/` is scientifically frozen.** Its algorithms reproduce the
  paper bit-for-bit and must not change without explicit sign-off. See the provenance tags
  in the [root README](../README.md).
- **`serving/` is the hardened, UI-facing adapter.** It is orchestration and IO only.
- **UI code must not implement scientific logic.** Species/background decisions,
  calibration, acquisition, and retirement all live in the frozen package. A UI collects
  labels and renders decisions; it does not compute them.
- **Dependency direction is one-way.** `serving/` may depend on `protorcd_pipeline/`;
  `protorcd_pipeline/` must **never** import from `serving/`. The science does not know the
  UI exists.

## The canonical round lifecycle

One review round, from the pool's point of view:

```
unresolved proposals
  -> acquire a batch          (density-moderated selection of what to review next)
  -> human review             (labels arrive via the LabelProvider)
  -> reviewed examples become prototypes   (ONLY human-reviewed labels enter the bank)
  -> triage the remaining proposals        (Resolve / Clear / Defer with the current bank)
  -> Resolve and Clear leave the pool PERMANENTLY  (retired; never re-reviewed, never
                                                    reused as supervision)
  -> only Defer continues to the next round
```

In code this whole round is a single call:

```python
info = session.step()      # acquire -> label -> triage -> retire
result = info.triage       # this round's Resolve / Clear / Defer over the remaining pool
```

### The invariant

Every round the kept pool (everything except the held-out calibration bank) is partitioned
into three disjoint sets whose union is the whole kept pool:

```
reviewed  ⊎  retired-auto  ⊎  remaining  ==  kept pool
```

Auto-decisions (Resolve/Clear) are **retired** — this is the single most important
correctness property of the loop, and the session asserts the partition every round. Only
human-reviewed labels ever become prototypes; automatic predictions are never fed back as
supervision.

## Where to go next

- Scientific method (the two inference problems, the decision rule, calibration,
  retirement, kept-space): [`protorcd_pipeline/triage/README.md`](../protorcd_pipeline/triage/README.md)
- The Python package by subpackage: [`protorcd_pipeline/README.md`](../protorcd_pipeline/README.md)
- How GT was removed vs the research code: [`gt_decoupling.md`](gt_decoupling.md)
- HTTP integration for a UI: [`serving/README.md`](../serving/README.md) + [`serving/API.md`](../serving/API.md)
- Reproducing the paper exactly: [`regression/README.md`](../regression/README.md)
- Runnable examples: [`examples/README.md`](../examples/README.md)
