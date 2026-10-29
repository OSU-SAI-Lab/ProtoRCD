# `serving` — HTTP service for the triage pipeline

> **UI developers, read in this order:** [`../docs/architecture.md`](../docs/architecture.md)
> (how the whole thing fits together and the layering rules) → this README (what the
> service does and its data-governance boundary) → [`API.md`](API.md) (the exact HTTP
> contract you build against). Current handoff reference: tag `serving-hardened-baseline`
> → `21fb490`.

Exposes the **validated** Resolve/Clear/Defer pipeline (scientific core frozen at
`08e0fa0`) to an external UI over HTTP/JSON. This layer is **orchestration and IO only** —
it never modifies the triage / acquisition / calibration algorithms. The same code that
passes the [research-equivalence regression](../regression/) runs under the UI.

## What stays on the backend

Proposal embeddings (feature stores) and proposal metadata are **OSC-resident** and never
sent to the client. **Source images remain OSC-resident and their filesystem paths are
never exposed to the client**; the backend extracts and serves only the requested proposal
crops. The UI receives only:

* proposal **ids** and their **bounding boxes**, and
* rendered **crop JPEGs** for the proposals it is reviewing (cut from the source image by
  the backend on request).

So crop *pixels* do reach the client (they must, for the reviewer to see them) — what is
protected is the original image files and the OSC directory structure, not the individual
crops the reviewer is shown. Feature-store paths and image roots are configured
server-side in a dataset registry; the UI only ever sends a dataset **name**.

## Architecture (why it's safe for the frozen baseline)

`TriageSession.build_calibration_bank()` and `step()` are **synchronous** — they call the
label provider inline and block until the reviewer answers. An HTTP request can't block
for minutes. So each session runs its whole lifecycle on a **worker thread**
([`session_manager.py`](session_manager.py)); whenever the loop needs labels it parks
inside a `QueueLabelProvider`, and the `POST /labels` endpoint delivers them and resumes
the worker.

The decision math is untouched — the worker thread is simply *where* the existing blocking
calls wait. `step()` (acquire → triage → **retire auto-decisions**) runs verbatim, so the
per-round partition invariant and research parity hold exactly as validated.

```
create session ─▶ worker parks on CALIBRATION batch
   POST /labels ─▶ calibrate ─▶ worker parks on REVIEW round 1
   POST /labels ─▶ triage + retire ─▶ worker parks on round 2
   ...            until finished or max_rounds
```

## Running

```bash
# 1. Copy the tracked schema template and fill in real paths for your deployment.
#    Keep the filled-in copy OUT of git (site-specific config, absolute paths).
cp serving/datasets.example.json my_datasets.json    # then edit my_datasets.json

# 2. Point the server at it and run.
export PROTORCD_SERVING_CONFIG=/abs/path/to/my_datasets.json
uvicorn serving.server:app --host 0.0.0.0 --port 8000
```

### Dataset registry

The registry maps a **logical dataset name** (all the client ever sends) to the backend
paths for that dataset. See [`datasets.example.json`](datasets.example.json) for the full
schema with placeholder paths and comments; the fields are:

| Field | Meaning |
|---|---|
| `name` | logical dataset id the client requests |
| `bg_features` / `fg_features` | DINOv3 / BioCLIP-2 feature-store dir(s); several = pooled multi-split, in order |
| `coco` | COCO annotation file(s); optional, only for offline-GT calibration/labels |
| `split_tags` | split names for a pooled multi-split dataset (aligns with the feature lists) |
| `image_root` | single image tree for crops (single-split datasets) |
| `image_roots` | `{split_tag: tree}` for pooled datasets whose per-split `file_name`s live under different trees (e.g. LoTE train/test) — set this **instead of** `image_root` |
| `labels` / `labels_file` | taxonomy (see below); optional |
| `density_cache` | frozen kept-space density `.npy` to pin exact tie-breaking on large pools; optional |

Do **not** commit a registry with real absolute paths — that is site-specific deployment
state. On the OSC deployment the filled-in registry lives at
`datasets_dryrun/serving_datasets.json` (gitignored), alongside a symlink layout that keeps
the actual dataset paths in one place.

`coco` is optional — it is only needed for offline-GT calibration/labels (reproducibility
mode). A live deployment where humans label everything can omit it.

The **taxonomy** (allowed foreground class ids) may be given inline as `labels`
(`{id: name}`), via `labels_file` (a JSON `{id: name}` file), or — if neither is set —
derived from the COCO categories. It drives server-side validation of submitted labels
(out-of-taxonomy / negative ids are rejected before they can enter the prototype bank) and
is exposed at `GET /datasets/{name}/labels`, so the frontend renders the class picker from
the server rather than hardcoding ids. If no taxonomy can be resolved, only label *sign*
is checked and a warning is logged.

## HTTP contract

See [`API.md`](API.md) for the full request/response shapes. In brief:

| Method & path | Purpose |
|---|---|
| `GET /datasets` | list configured dataset names |
| `GET /datasets/{name}/labels` | the dataset's label taxonomy (class ids the server accepts) |
| `POST /sessions` | create + start a session (returns the first pending batch + taxonomy) |
| `GET /sessions/{id}` | session state (round, reviews, remaining, rule, pending batch) |
| `GET /sessions/{id}/next-batch` | the batch currently awaiting labels |
| `POST /sessions/{id}/labels` | submit labels for the pending batch; advances one phase |
| `GET /sessions/{id}/triage` | the latest round's Resolve/Clear/Defer decisions |
| `GET /crops/{id}/{proposal_id}` | JPEG crop of a proposal |
| `DELETE /sessions/{id}` | discard a session |

## Labels

A label is an integer per proposal: **`0` == background**, any positive integer is a
species class id (the same class-id space as the dataset's COCO categories / taxonomy). The
UI shows the reviewer each proposal's crop and collects one label per proposal in the
batch. Positive ids are validated against the dataset taxonomy server-side, so a UI bug
cannot inject an arbitrary class into the prototype bank.

## Constraints for future work

* **Do not** change `protorcd_pipeline/triage/*` (the frozen algorithms) while building the
  UI. Add orchestration here, not decision logic.
* After any backend change, re-run `regression/run_all.sh` and confirm **ALL REGRESSIONS
  PASSED** (10/10). See [../regression/README.md](../regression/README.md).

## Status / not yet built

Sessions are held **in memory** (no persistence across a server restart) and there is **no
authentication**. Both are deliberate: a UI team can add a session store and an auth layer
around this contract without touching the pipeline. Multi-worker deployment needs a shared
session store (the in-memory manager assumes one process).
