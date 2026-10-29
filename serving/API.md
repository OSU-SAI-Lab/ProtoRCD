# Triage service — HTTP API contract

Base URL: wherever `uvicorn serving.server:app` is served (e.g. `http://host:8000`).
All request/response bodies are JSON unless noted. A label is an int: `0` == background,
positive == species class id. Positive ids must belong to the dataset's taxonomy (see
`GET /datasets/{dataset}/labels`); out-of-taxonomy or negative ids are rejected with HTTP
`400` and never enter the prototype bank.

The typical UI flow is a loop of `next-batch` (or the batch returned by the previous
call) → render crops → collect labels → `POST /labels`, until `finished` is true.

---

## `GET /datasets`

List the dataset names the server is configured to serve.

```json
{ "datasets": ["ENA24", "LoTE"] }
```

---

## `GET /datasets/{dataset}/labels`

The dataset's label taxonomy — the class ids the server will accept. Use it to render the
reviewer's class picker instead of hardcoding ids in the frontend.

```json
{
  "dataset": "ENA24",
  "background_id": 0,
  "validated": true,
  "labels": { "1": "American Black Bear", "2": "White-tailed Deer", "...": "..." }
}
```

`background_id` is always `0`. If `validated` is `true`, `POST .../labels` rejects any
positive id not in `labels` (and any negative id) with HTTP `400`. If `validated` is
`false` (`labels` is `null` — no taxonomy configured for the dataset), only the sign is
checked. The same taxonomy is returned as the `labels` field of `POST /sessions`.

---

## `POST /sessions`

Create a session and start its worker. Returns once the **first batch** (calibration,
unless `offline_calibration` short-circuits it) has been acquired and is awaiting labels.

Request:

```json
{
  "dataset": "ENA24",
  "seed": 3,
  "batch_size": 50,
  "calibration_size": 100,
  "max_rounds": 16,
  "offline_calibration": false
}
```

* `offline_calibration` (default `false`): if `true` and the dataset has COCO configured,
  the calibration batch is auto-answered from ground truth server-side, so the first
  batch returned is review **round 1**. Use for reproducibility/demo; leave `false` when a
  human should label the calibration bank.

Response:

```json
{
  "session_id": "b1c2...hex",
  "state": { ...see GET /sessions/{id}... },
  "labels": { "1": "American Black Bear", "...": "..." },  // taxonomy, or null if none
  "next_batch": {
    "batch_id": "b1c2...:calibration:0:1",   // echo this back on submit
    "phase": "calibration",           // or "review"
    "round": 0,                       // 0 during calibration, else the review round
    "proposals": [
      { "proposal_id": "1014:0", "image_id": 1014, "bbox_xywh": [x, y, w, h] }
    ]
  }
}
```

`next_batch` is `null` only if the session finished immediately (degenerate pools).

---

## `GET /sessions/{id}`

```json
{
  "session_id": "b1c2...",
  "status": "awaiting_labels",
  "error": null,
  "calibrated": true,
  "rule": { "k": 7, "tau_fg": 5, "tau_bg": 5, "tau_veto": 8,
            "support_quantile": 0.0, "foreground_first": true },
  "round": 3,
  "cumulative_reviews": 150,
  "remaining": 10500,
  "finished": false,
  "failed": false,
  "pending_batch": { "batch_id": "b1c2...:review:4:8", "phase": "review", "round": 4, "size": 50 }
}
```

`status` is one of `starting | awaiting_labels | advancing | finished | failed`. On
`failed`, `error` contains the exception **type name only** (e.g. `"FileNotFoundError"`);
the detailed message is retained in the server logs and is not exposed to the client, and
every action-taking call returns HTTP 500 rather than blocking. `rule` is `null` before
calibration completes. `pending_batch` is `null`
when no batch is awaiting labels.

---

## `GET /sessions/{id}/next-batch`

The batch currently awaiting labels. Blocks briefly if acquisition is still running; never
blocks for human-review duration.

```json
{
  "batch_id": "b1c2...:review:4:8",
  "phase": "review",
  "round": 4,
  "finished": false,
  "proposals": [
    { "proposal_id": "348:0", "image_id": 348, "bbox_xywh": [10, 20, 64, 48] }
  ]
}
```

When the session is done: `{ "finished": true, "proposals": [] }`. If the worker failed,
returns HTTP 500 with a generic client-facing message (not a hang); the detailed cause is
logged server-side.

---

## `POST /sessions/{id}/labels`

Submit labels for the **pending** batch. The server applies them (calibrating, or running
triage + retirement for a review round), then returns that round's triage result and the
**next** batch in one round-trip.

Request:

```json
{
  "batch_id": "b1c2...:review:4:8",
  "labels": { "348:0": 0, "1014:0": 7, "77:1": 3 }
}
```

Every proposal id in the pending batch must be present. Keys are proposal ids as strings.
`batch_id` should be the id of the batch being answered (from `next-batch` /
`next_batch`); if it does not match the current pending batch the request is rejected with
HTTP `409` and **no labels are consumed** — this makes submits idempotent and stale-safe
against a duplicate post, a stale browser tab, or a second client.

Response:

```json
{
  "state": { ...GET /sessions/{id}... },
  "triage": {
    "round": 4,
    "retired": 812,
    "counts": { "resolve": 500, "clear": 312, "defer": 9738 },
    "decisions": [
      { "proposal_id": "3000:0", "decision": "defer",   "predicted_class": 0 },
      { "proposal_id": "5060:2", "decision": "resolve", "predicted_class": 7 },
      { "proposal_id": "8706:0", "decision": "clear",   "predicted_class": 0 }
    ]
  },
  "next_batch": { "phase": "review", "round": 5, "proposals": [ ... ] }
}
```

* `triage` is `null` for the calibration phase (no triage happens then).
* `retired` = number of proposals auto-decided (Resolve + Clear) this round; they leave
  the pool and are never shown again.
* `predicted_class` is the species id for `resolve`, `0` otherwise.
* `next_batch` is `null` once the session is finished.

---

## `GET /sessions/{id}/triage`

The most recent completed round's triage result (same shape as the `triage` field above),
or `{ "round": R, "triage": null }` if no round has completed yet.

---

## `GET /crops/{id}/{proposal_id}`

Returns the proposal's crop as `image/jpeg` bytes (backend cuts `bbox_xywh` from the
source image and downscales the longer edge to a sane size). By default the crop is the
tight bounding box — no context padding — configurable server-side via `CropService`. Use
it directly as an `<img src>`.

`404` if the session or proposal is unknown, or the source image is unavailable (the
resolved path is logged server-side, never returned to the client).

---

## `DELETE /sessions/{id}`

Discard a session and free its resources.

```json
{ "deleted": "b1c2..." }
```

---

## Errors

Standard HTTP status codes with `{ "detail": "..." }` bodies:

* `404` unknown dataset / session / proposal.
* `400` labels missing for pending proposals, or a label id is negative / not in the
  dataset taxonomy. No labels are consumed; fix and resubmit the same batch.
* `409` stale/duplicate label submission (`batch_id` does not match the current pending
  batch, or no batch is awaiting labels). No labels are consumed; re-fetch the current
  batch and resubmit.
* `500` the session worker failed (status `failed`) or another backend error — a generic
  client-facing message; the detailed cause is logged server-side. A failed session stays
  failed; create a new session.
* `504` a batch/phase did not become ready within the server's wait window (rare). The
  correct client response depends on the endpoint:
  * on `GET .../next-batch` — nothing was consumed; retry / poll.
  * on `POST .../labels` — **the labels were already accepted and the batch consumed**;
    do **not** resubmit (a resubmit would get `409`). Poll `GET /sessions/{id}` until
    `status` leaves `advancing`.
