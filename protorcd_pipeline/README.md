# `protorcd_pipeline` — the scientific package

The three-stage detect → features → triage pipeline. This package is the **scientific**
side of the repo, and it must not depend on `serving/` or on any UI. The frozen **triage**
implementation is validated to reproduce the paper's research trajectories bit-for-bit
*given the corresponding proposal and feature stores* — the equivalence harness holds the
stores fixed; it does not claim SAM3 detection or feature extraction are bit-for-bit
deterministic across hardware. For the big picture and the layering rules, see
[`../docs/architecture.md`](../docs/architecture.md).

Each subpackage below lists its input, its output, and whether it is part of the frozen
scientific baseline (tag `scientific-parity-baseline` → `08e0fa0`; see the
[root README](../README.md)).

## `detect/` — stage 1: proposals

SAM3 text-prompt detection that turns imagery into **class-agnostic** animal proposals.

- **Input:** images (+ a COCO-style image list).
- **Output:** raw SAM3 detection JSON (boxes/masks per image).
- **Entry point:** `python -m protorcd_pipeline.detect.run_sam3_text_detection` (needs the
  `detect` extra — torch/SAM3).
- **Frozen scientific baseline:** yes.

## `features/` — stage 2: embeddings

Embeds each proposal with **two** frozen backbones. Row `i` of the two stores is the same
proposal, in the same order.

- **Input:** standardized proposals + the source images.
- **Output:** two *feature stores* (one DINOv3 = background evidence, one BioCLIP-2 =
  foreground/species evidence). A store is a directory with `manifest.json`, `chunk_*.npz`
  (key `embeddings`), and `metadata.jsonl` (one record per proposal: `proposal_id`,
  `image_id`, `file_name`, `bbox_xywh`).
- **Entry point:** `ExtractionConfig` + `extract_embeddings()` in
  [`features/extract.py`](features/extract.py) (needs the `features` extra).
- **Frozen scientific baseline:** yes.

## `common/` — detect → features bridge + shared utilities

- **Input / output:** converts raw SAM3 detection JSON into the standardized proposal
  format the feature stage consumes (`sam3_to_proposals.py`); shared record types,
  geometry, and IO helpers (`records.py`, `geometry.py`, `io.py`).
- **Frozen scientific baseline:** yes.

## `triage/` — stage 3: active Resolve / Clear / Defer

The active-review loop and the decision rule. This is the scientifically load-bearing part.

- **Input:** two aligned feature stores (from stage 2) + human labels supplied through a
  `LabelProvider`.
- **Output:** per-proposal Resolve / Clear / Defer decisions; a growing human-reviewed
  prototype bank; the calibrated rule.
- **Entry point:** `TriageSession.for_protocol("split_no_veto", pool, provider, ...)`.
- **Depends only on** numpy + scikit-learn (no torch), so it can run next to a lightweight
  UI while detection/feature extraction run elsewhere.
- **Frozen scientific baseline:** yes — this is the algorithm the equivalence regression
  guards.

**For the algorithm itself** — the two inference problems, the decision rule, the canonical
`split_no_veto` protocol, calibration, retirement, and kept-space indexing — read
[`triage/README.md`](triage/README.md).
