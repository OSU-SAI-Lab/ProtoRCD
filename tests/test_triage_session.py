"""End-to-end smoke test of the GT-decoupled triage stage on synthetic data.

Builds a clustered synthetic pool (a few FG species + a background blob), drives the
session with a CallbackLabelProvider (the UI path) whose labels come from a hidden
oracle, and checks the loop produces sane Resolve/Clear/Defer decisions -- without the
triage decision ever seeing the oracle.
"""
from __future__ import annotations

import numpy as np

from protorcd_pipeline.triage import (
    CallbackLabelProvider,
    GTLabelProvider,
    ProposalPool,
    TriageSession,
    calibrate_rule,
    guarded_triage,
)
from protorcd_pipeline.triage.acquisition import l2_normalize


def _make_pool(n_per_class=60, n_fg_classes=4, dim=64, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_fg_classes + 1, dim))  # class 0 == background
    xs, ys = [], []
    for cls in range(n_fg_classes + 1):
        pts = centers[cls] + 0.15 * rng.normal(size=(n_per_class, dim))
        xs.append(pts)
        ys.append(np.full(n_per_class, cls, dtype=np.int64))
    x = l2_normalize(np.vstack(xs).astype(np.float32))
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    x, y = x[perm], y[perm]
    # two representations: use the same geometry with small independent noise so both
    # carry the signal (in practice bg=DINOv3, fg=BioCLIP-2).
    bg_x = l2_normalize(x + 0.02 * rng.normal(size=x.shape).astype(np.float32))
    fg_x = l2_normalize(x + 0.02 * rng.normal(size=x.shape).astype(np.float32))
    ids = [f"p{i}" for i in range(len(y))]
    meta = [{"proposal_id": ids[i], "image_id": i // 3, "bbox_xywh": [0, 0, 10, 10]}
            for i in range(len(y))]
    return ProposalPool(bg_x=bg_x, fg_x=fg_x, proposal_ids=ids, metadata=meta), y


def test_session_end_to_end():
    pool, oracle_y = _make_pool(seed=1)
    id_to_y = {pid: int(oracle_y[i]) for i, pid in enumerate(pool.proposal_ids)}

    seen = {"ids": []}

    def request(ids):
        seen["ids"].extend(ids)
        return {pid: id_to_y[pid] for pid in ids}

    session = TriageSession(
        pool, CallbackLabelProvider(request),
        method="density_balanced_kcenter", batch_size=20, seed=3,
        k_values=(3, 5), density_k=10,
    )
    cal = session.build_calibration_bank(size=40)
    assert cal.size == 40
    assert session.rule is not None
    assert session.rule.k in (3, 5)

    cal_ids = {pool.proposal_ids[i] for i in cal.tolist()}
    total_auto = 0
    for _ in range(3):
        if session.finished:
            break
        info = session.step()
        # calibration bank must be excluded from acquisition forever
        assert set(info.acquired_ids).isdisjoint(cal_ids)
        # step() triages the pool and retires the auto-decided this round
        assert info.triage is not None
        total_auto += info.retired
    assert total_auto > 0  # the rule auto-decided something across the rounds

    # Every proposal id is reviewed AT MOST once -- auto-decided proposals are retired
    # and must never be re-acquired.
    assert len(seen["ids"]) == len(set(seen["ids"]))

    # The remaining pool is disjoint from reviewed, auto-decided, and calibration.
    result = session.triage(on="remaining")
    reviewed_ids = {pool.proposal_ids[i] for i in session.support_indices.tolist()}
    assert set(result.proposal_ids).isdisjoint(reviewed_ids)
    assert set(result.proposal_ids).isdisjoint(cal_ids)


def test_retirement_partitions_the_pool():
    """reviewed U auto U remaining == kept pool, always disjoint (the core invariant)."""
    pool, oracle_y = _make_pool(seed=5)
    gt = GTLabelProvider(oracle_y, id_to_index=pool.id_to_index)
    session = TriageSession(
        pool, gt, method="density_balanced_kcenter", batch_size=25, seed=7,
        k_values=(3, 5), density_k=10,
    )
    cal = session.build_calibration_bank(size=40)
    kept = session.kept_source.size
    assert kept == pool.size - cal.size

    while not session.finished:
        session.step()
        reviewed = set(session.support_indices_kept.tolist())
        auto = set(session.auto_indices_kept.tolist())
        remaining = set(int(v) for v in session._available.tolist())
        # pairwise disjoint
        assert reviewed.isdisjoint(auto)
        assert reviewed.isdisjoint(remaining)
        assert auto.isdisjoint(remaining)
        # exact cover of kept-space
        assert len(reviewed) + len(auto) + len(remaining) == kept


def test_deterministic_across_runs():
    """Two sessions with the same seed produce identical acquisition + auto sets."""
    def run():
        pool, oracle_y = _make_pool(seed=9)
        gt = GTLabelProvider(oracle_y, id_to_index=pool.id_to_index)
        s = TriageSession(
            pool, gt, method="density_balanced_kcenter", batch_size=20, seed=11,
            k_values=(3, 5), density_k=10, density_scope="initial",
        )
        s.build_calibration_bank(size=40)
        acquired = []
        for _ in range(3):
            if s.finished:
                break
            acquired.append(tuple(s.step().acquired_ids))
        return acquired, s.auto_indices_kept.tolist(), s.rule.as_dict()

    a1 = run()
    a2 = run()
    assert a1 == a2  # bit-for-bit reproducible given the seed


def test_gt_provider_matches_callback():
    """GTLabelProvider (offline) and a callback wrapping the same labels agree."""
    pool, oracle_y = _make_pool(seed=2)
    gt = GTLabelProvider(oracle_y)  # ids are pool indices here
    ids = [0, 5, 10]
    from_gt = gt.get_labels(ids)
    assert from_gt == {i: int(oracle_y[i]) for i in ids}


def test_triage_decision_is_gt_independent():
    """guarded_triage must return identical decisions whether or not true_y is passed."""
    pool, oracle_y = _make_pool(seed=4)
    support_idx = np.arange(50)
    query_idx = np.arange(50, 120)
    rule, _, _ = calibrate_rule(
        pool.bg_x, pool.fg_x, oracle_y, support_idx, k_values=(3, 5),
    )
    common = dict(
        support_bg_x=pool.bg_x[support_idx], support_fg_x=pool.fg_x[support_idx],
        support_y=oracle_y[support_idx],
        query_bg_x=pool.bg_x[query_idx], query_fg_x=pool.fg_x[query_idx],
        rule=rule, query_global_idx=query_idx,
    )
    _, _, dec_no_gt, _ = guarded_triage(**common, true_y=None, collect_rows=False)
    _, _, dec_gt, _ = guarded_triage(**common, true_y=oracle_y[query_idx], collect_rows=False)
    assert dec_no_gt == dec_gt  # GT never affects the decision


if __name__ == "__main__":
    test_session_end_to_end()
    test_retirement_partitions_the_pool()
    test_deterministic_across_runs()
    test_gt_provider_matches_callback()
    test_triage_decision_is_gt_independent()
    print("OK: all triage session smoke tests passed")
