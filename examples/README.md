# Examples — three ways to run the triage loop

All three drive the **same** frozen `TriageSession` via
`for_protocol("split_no_veto", ...)`; they differ only in **where the human labels come
from**. For the layering picture see [`../docs/architecture.md`](../docs/architecture.md).

The UI is **optional**. The pipeline must always remain runnable headlessly — the offline
path below is the reference way to run and reproduce it with no UI at all.

## 1. `run_triage_offline.py` — headless GT / reproducibility mode

Labels come from an offline `GTLabelProvider` (IoU-matching proposals to COCO ground
truth), so the whole loop runs unattended. Use it to sanity-check a pool or to reproduce
the paper trajectory.

```bash
python examples/run_triage_offline.py \
    --bg-features /path/to/dinov3_store \
    --fg-features /path/to/bioclip_store \
    --coco        /path/to/annotations.json \
    --batch-size 50 --rounds 16 --seed 3
```

Each `session.step()` is a full round (acquire → label → triage → retire); the script
prints per-round Resolve/Clear/Defer counts and auto-coverage.

## 2. `ui_integration.py` — direct provider examples (in-process)

Shows a Python UI driving the loop directly, with two label-provider patterns:

- **`CallbackLabelProvider`** — the UI implements `request(ids) -> {id: label}` in-process
  (e.g. Gradio/Streamlit). Note how the stub rejects an unknown label choice instead of
  silently mapping it to background.
- **`FileLabelProvider`** — an out-of-process UI exchanges per-round JSON files.

This is reference code, not a runnable app.

## 3. `serving/` — recommended external / browser UI

For a real front-end, use the HTTP service in [`../serving/`](../serving/README.md) rather
than embedding the session in the UI process. It keeps proposal data and images on the
backend and hands the client only ids, boxes, and rendered crops. Build against
[`../serving/API.md`](../serving/API.md).

## Which one should I use?

| Goal | Path |
|---|---|
| Reproduce paper numbers / smoke-test a pool | `run_triage_offline.py` |
| Embed the loop in a Python UI | `ui_integration.py` patterns |
| Build a browser / remote UI | the `serving/` HTTP API |
