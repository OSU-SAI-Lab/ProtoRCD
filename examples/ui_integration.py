#!/usr/bin/env python3
"""How a UI drives the triage loop -- the primary (non-GT) integration paths.

For an EXTERNAL / browser UI, prefer the HTTP service in ``serving/`` (see
``serving/README.md`` + ``serving/API.md``): it wraps exactly this loop, keeps proposal
data and images on the backend, and serves crops. The two in-process patterns below are
for a Python UI (e.g. Gradio/Streamlit) that runs the session directly.

1. In-process (CallbackLabelProvider): the UI implements
   ``request(proposal_ids) -> {id: label}`` by showing the reviewer the corresponding
   proposal crops and collecting their choices.

2. Out-of-process (FileLabelProvider): the session runs as one process (e.g. on a GPU
   node) and a separate UI exchanges JSON files. The session writes
   ``round_<r>_query.json``; the UI writes ``round_<r>_labels.json``.

Both are pure functions of the two feature stores from stage 2 -- no ground truth. Use
``TriageSession.for_protocol("split_no_veto", ...)`` to run the validated paper protocol.

The proposal id -> crop mapping the UI needs is in ``pool.metadata[i]`` (image_id,
file_name, bbox_xywh), keyed by ``pool.proposal_ids[i]``.
"""
from __future__ import annotations

from typing import Dict, Sequence

from protorcd_pipeline.triage import (
    CallbackLabelProvider,
    FileLabelProvider,
    TriageSession,
    load_pool,
)


# --------------------------------------------------------------------------- #
# Pattern 1: in-process callback                                              #
# --------------------------------------------------------------------------- #
def run_in_process(bg_features: str, fg_features: str, *, species_to_id: Dict[str, int]):
    pool = load_pool(bg_features, fg_features)

    # Map proposal_id -> crop location the UI will render.
    crop_of = {
        pid: pool.metadata[i]  # {"image_id", "file_name", "bbox_xywh", ...}
        for i, pid in enumerate(pool.proposal_ids)
    }

    def request(proposal_ids: Sequence) -> Dict:
        """The UI implements THIS. Show each proposal's crop, collect a label.

        Return an int per id: 0 == background, else the species id. Below is a stub
        that a real UI replaces with human input.
        """
        labels = {}
        for pid in proposal_ids:
            meta = crop_of[pid]
            # your_ui.show(meta["file_name"], meta["bbox_xywh"])
            # choice = your_ui.await_reviewer_choice()   # e.g. "background" or a species
            choice = "background"  # <-- stub; a real UI returns the reviewer's choice
            # Map the reviewer's choice to a class id. An UNKNOWN choice must NOT silently
            # become background (0) -- that would poison the support set with a mislabel.
            # Reject it explicitly so the bug surfaces at the source.
            if choice != "background" and choice not in species_to_id:
                raise ValueError(
                    f"unknown label choice {choice!r} for proposal {pid!r}; "
                    f"expected 'background' or one of {sorted(species_to_id)}"
                )
            labels[pid] = 0 if choice == "background" else species_to_id[choice]
        return labels

    # Validated paper protocol (foreground_first, veto off, supermajority calibration,
    # density-moderated k-center, initial-scope density).
    session = TriageSession.for_protocol(
        "split_no_veto", pool, CallbackLabelProvider(request), seed=3, batch_size=50,
    )
    session.build_calibration_bank(size=100)   # reviews the bank via request(), calibrates

    while not session.finished:
        info = session.step()                  # acquire -> request() -> triage -> RETIRE
        result = info.triage                   # this round's decisions over the pool
        # your_ui.render_triage(result.proposal_ids, result.decisions, result.predicted_class)
        if session.cumulative_reviews >= 800:   # stop when the review budget is spent
            break
    return session


# --------------------------------------------------------------------------- #
# Pattern 2: out-of-process file exchange                                     #
# --------------------------------------------------------------------------- #
def run_out_of_process(bg_features: str, fg_features: str, exchange_dir: str):
    pool = load_pool(bg_features, fg_features)
    provider = FileLabelProvider(exchange_dir, poll_seconds=2.0)

    session = TriageSession.for_protocol(
        "split_no_veto", pool, provider, seed=3, batch_size=50,
    )
    session.build_calibration_bank(size=100)
    # Each step() writes round_<r>_query.json and blocks until the UI writes
    # round_<r>_labels.json. The UI polls the exchange dir, renders the crops for the
    # queued proposal_ids, and writes back {"labels": {proposal_id: label}}. step() then
    # triages the rest and retires the auto-decided; info.triage holds those decisions.
    while not session.finished:
        info = session.step()
        _ = info.triage  # render / persist the round's Resolve/Clear/Defer for the UI
        if session.cumulative_reviews >= 800:
            break
    return session


if __name__ == "__main__":
    print(__doc__)
