# ProtoRCD Pipeline

A clean, UI-integratable pipeline for ecological camera-trap object detection with active
**prototype triage** — Resolve / Clear / Defer over detector proposals, decoupled from
ground-truth boxes so a UI can drive review on live proposals.

## The three-stage pipeline

```
imagery
  -> SAM3 proposals                    (stage 1: class-agnostic animal boxes)
  -> DINOv3 + BioCLIP-2 feature stores (stage 2: two aligned embeddings per proposal)
  -> TriageSession -> Resolve/Clear/Defer   (stage 3: active review + auto-decision)
```

**DINOv3** carries background/validity evidence ("is this a real animal box?"),
**BioCLIP-2** carries foreground/species evidence ("which species?"). Stage 3 is
**decoupled from ground truth**: labels arrive through a `LabelProvider` (a UI callback, a
JSON file exchange, or — for reproducing paper numbers — an offline GT provider); the
triage *decision* never reads ground truth.

For the conceptual walkthrough and the layering rules, start with
[`docs/architecture.md`](docs/architecture.md).

## Frozen provenance tags (read before changing anything)

Three tagged commits are the authoritative baselines. **Tags are immutable provenance
markers — never move them;** each validated stage gets a new tag.

| Tag | Commit | What it is |
|---|---|---|
| `scientific-parity-baseline` | `08e0fa0` | The frozen scientific implementation, with the triage / acquisition / calibration path validated to reproduce the research runs **bit-for-bit** given the corresponding proposal and feature stores (70/70 checks, 10/10 runs; see [regression/](regression/README.md)). |
| `serving-baseline` | `247d741` | First HTTP service exposing the pipeline to an external UI. |
| `serving-hardened-baseline` | `21fb490` | **Current handoff point.** Adds serving-layer hardening (atomic submit, mandatory `batch_id`, path sanitization, taxonomy validation). `protorcd_pipeline/` byte-identical to `08e0fa0`. |

## Where to start

| You are… | Start here |
|---|---|
| a **UI developer** | [`docs/architecture.md`](docs/architecture.md) → [`serving/README.md`](serving/README.md) → [`serving/API.md`](serving/API.md) |
| running it **headless** / reproducing results | [`examples/README.md`](examples/README.md) + [`regression/README.md`](regression/README.md) |
| working on the **science** | [`protorcd_pipeline/README.md`](protorcd_pipeline/README.md) → [`protorcd_pipeline/triage/README.md`](protorcd_pipeline/triage/README.md) |

## Repository map

```text
docs/
  architecture.md          # conceptual onboarding + layering rules (start here)
  gt_decoupling.md         # how GT was removed vs the research code
protorcd_pipeline/           # the scientific package (frozen at 08e0fa0)
  detect/                  #   stage 1: SAM3 text-prompt detection
  features/                #   stage 2: DINOv3 + BioCLIP-2 embeddings
  common/                  #   detect->features bridge + shared utils
  triage/                  #   stage 3: acquisition + Resolve/Clear/Defer
serving/                   # hardened HTTP adapter for an external UI
  API.md                   #   the HTTP contract the UI builds against
examples/                  # headless + in-process provider examples
tests/                     # fast unit/integration tests
regression/                # bit-for-bit research-equivalence harness
```

## Install

```bash
pip install -e .                    # stage 3 only (numpy + scikit-learn) — triage next to a UI
pip install -e ".[serving]"         # + the HTTP service (FastAPI/uvicorn/pydantic/pillow)
pip install -e ".[detect,features]" # + detection & feature extraction (heavy ML stack)
pip install -e ".[serving,dev]"     # everything for development + tests
```

Stage 3 has **no torch dependency**, so a lightweight UI process can run the
acquisition/triage loop while a separate GPU job does detection + feature extraction.

## Quick start

### Headless (no UI)

The offline path drives the full loop with GT labels — the reference way to run and
reproduce it. The canonical entry point is `for_protocol("split_no_veto")`, and one
`step()` is a complete round (acquire → label → triage → retire):

```python
from protorcd_pipeline.triage import TriageSession, GTLabelProvider, load_pool, labels_from_coco

pool = load_pool("dinov3_store/", "bioclip_store/")     # background_dir, foreground_dir
gt, _ = labels_from_coco(pool.metadata, "annotations.json")
provider = GTLabelProvider(gt, id_to_index=pool.id_to_index)

session = TriageSession.for_protocol("split_no_veto", pool, provider, seed=3, batch_size=50)
session.build_calibration_bank(size=100)
while not session.finished:
    info = session.step()          # acquire -> label -> triage -> retire auto-decisions
    result = info.triage           # this round's Resolve / Clear / Defer over the pool
```

Runnable version: [`examples/run_triage_offline.py`](examples/run_triage_offline.py). Swap
`GTLabelProvider` for `CallbackLabelProvider`/`FileLabelProvider` to plug in a UI — see
[`examples/README.md`](examples/README.md).

### HTTP / UI

Run the service and build a front-end against its contract; proposal data and images stay
on the backend.

```bash
export PROTORCD_SERVING_CONFIG=/path/to/serving_datasets.json
uvicorn serving.server:app --host 0.0.0.0 --port 8000
```

See [`serving/README.md`](serving/README.md) and [`serving/API.md`](serving/API.md).

## Tests & regression

```bash
pip install -e ".[serving,dev]"   # makes protorcd_pipeline + serving importable
pytest tests/                     # 10/10 (triage + serving)
# fresh clone without installing: PYTHONPATH=. pytest tests/

regression/run_all.sh             # frozen research-equivalence sweep -> ALL REGRESSIONS PASSED (10/10)
```

The equivalence sweep is heavier and separate; details and the density caveat are in
[`regression/README.md`](regression/README.md).

## Contribution rules

- **Do not modify `protorcd_pipeline/`** (the scientific pipeline). It is frozen at
  `08e0fa0` and guarded by the equivalence regression.
- **UI code builds only against [`serving/API.md`](serving/API.md)** — the client sends a
  logical dataset name and never needs OSC paths. UI code must not implement scientific
  logic.
- **Dependency direction is one-way:** `serving/` may import `protorcd_pipeline/`;
  `protorcd_pipeline/` must never import `serving/`.
- Any change that touches the backend must keep **both** green before it lands:
  `pytest tests/` (10/10) and `regression/run_all.sh` (10/10). If it can't, it's out of
  scope until discussed.
