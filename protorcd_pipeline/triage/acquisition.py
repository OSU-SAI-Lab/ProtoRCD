"""Active-acquisition policies over proposal embeddings (stage 3, part 1).

These functions are pure geometry over L2-normalized embeddings and have **no
dependency on ground-truth boxes or labels** -- they only decide *which* proposals
to send for human review next. The canonical paper policy is
``density_balanced_kcenter`` ("density-moderated"): a farthest-point (k-center)
selection weighted by ``q * (1 - q)`` where ``q`` is the local-density percentile,
which biases acquisition toward intermediate-density regions.

Ported verbatim (logic-preserving) from the research code
``evaluate_active_triage_features.py`` so acquisition order matches the paper.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "l2_normalize",
    "approximate_density",
    "full_knn_percentile_density",
    "compute_density",
    "density_percentile_ranks",
    "select_random",
    "select_kcenter",
    "select_weighted_kcenter",
    "acquire",
    "ACQUISITION_METHODS",
]

ACQUISITION_METHODS = (
    "random",
    "kcenter",
    "density_kcenter",
    "density_balanced_kcenter",  # canonical "density-moderated" policy
)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, 1e-12)


# --------------------------------------------------------------------------- #
# Local density estimates                                                      #
# --------------------------------------------------------------------------- #
def approximate_density(x: np.ndarray, k: int, anchor_count: int, seed: int) -> np.ndarray:
    """Fast min-max-scaled density via mean cosine similarity to k nearest anchors."""
    n = x.shape[0]
    anchor_count = min(max(int(k) + 1, int(anchor_count)), n)
    rng = np.random.default_rng(seed)
    anchors = rng.choice(n, size=anchor_count, replace=False)
    sims = x @ x[anchors].T
    kk = min(int(k), sims.shape[1])
    top = np.partition(sims, kth=sims.shape[1] - kk, axis=1)[:, -kk:]
    density = np.mean(top, axis=1)
    lo, hi = float(np.min(density)), float(np.max(density))
    if hi <= lo:
        return np.ones((n,), dtype=np.float32)
    return ((density - lo) / (hi - lo)).astype(np.float32)


def full_knn_percentile_density(x: np.ndarray, k: int, block_size: int = 512) -> np.ndarray:
    """Exact local density as percentile-ranked mean kNN cosine similarity."""
    n = x.shape[0]
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    kk = max(1, min(int(k), n - 1))
    block_size = max(1, int(block_size))
    density = np.empty((n,), dtype=np.float32)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        sims = (x[start:stop] @ x.T).astype(np.float32, copy=False)
        rows = np.arange(stop - start)
        sims[rows, np.arange(start, stop)] = -np.inf
        top = np.partition(sims, kth=n - kk, axis=1)[:, -kk:]
        density[start:stop] = np.mean(top, axis=1)

    return density_percentile_ranks(density)


def compute_density(
    x: np.ndarray,
    mode: str,
    k: int,
    anchor_count: int,
    seed: int,
    block_size: int = 512,
) -> np.ndarray:
    if mode == "anchor":
        return approximate_density(x, k=k, anchor_count=anchor_count, seed=seed)
    if mode == "full_knn":
        return full_knn_percentile_density(x, k=k, block_size=block_size)
    raise ValueError(f"Unsupported density mode: {mode}")


def density_percentile_ranks(density: np.ndarray) -> np.ndarray:
    order = np.argsort(density)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(density), dtype=np.float32)
    return ranks / float(max(1, len(density) - 1))


# --------------------------------------------------------------------------- #
# Selection policies                                                           #
# --------------------------------------------------------------------------- #
def select_random(n: int, budget: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(n, size=min(budget, n), replace=False).astype(np.int64)


def select_kcenter(x: np.ndarray, budget: int) -> np.ndarray:
    """Unweighted farthest-point (k-center greedy) selection in cosine space.

    The seed point is the proposal nearest the pool centroid (matching the research
    ``select_kcenter``), so the greedy trajectory is deterministic and reproducible.
    """
    budget = min(int(budget), x.shape[0])
    if budget <= 0:
        return np.asarray([], dtype=np.int64)
    centroid = l2_normalize(x.mean(axis=0, keepdims=True))[0]
    first = int(np.argmax(x @ centroid))
    selected = [first]
    mask = np.zeros(x.shape[0], dtype=bool)
    mask[first] = True
    max_sim = x @ x[first]
    while len(selected) < budget:
        scores = 1.0 - max_sim
        scores[mask] = -np.inf
        nxt = int(np.argmax(scores))
        selected.append(nxt)
        mask[nxt] = True
        max_sim = np.maximum(max_sim, x @ x[nxt])
    return np.asarray(selected, dtype=np.int64)


def select_weighted_kcenter(x: np.ndarray, budget: int, weights: np.ndarray) -> np.ndarray:
    """Farthest-point selection where the marginal gain is scaled by ``weights``."""
    weights = np.maximum(np.asarray(weights, dtype=np.float32), 1e-6)
    budget = min(int(budget), x.shape[0])
    if budget <= 0:
        return np.asarray([], dtype=np.int64)
    selected = [int(np.argmax(weights))]
    mask = np.zeros(x.shape[0], dtype=bool)
    mask[selected[0]] = True
    max_sim = x @ x[selected[0]]
    while len(selected) < budget:
        scores = (1.0 - max_sim) * weights
        scores[mask] = -np.inf
        nxt = int(np.argmax(scores))
        selected.append(nxt)
        mask[nxt] = True
        max_sim = np.maximum(max_sim, x @ x[nxt])
    return np.asarray(selected, dtype=np.int64)


def acquire(method: str, x: np.ndarray, budget: int, density: np.ndarray, seed: int) -> np.ndarray:
    """Return indices (into ``x``) of the ``budget`` proposals to review next.

    ``x``       L2-normalized embeddings of the *currently unlabeled* pool.
    ``density`` local-density estimate for the same pool (from ``compute_density``);
                ignored by ``random``/``kcenter``.
    """
    if method == "random":
        return select_random(x.shape[0], budget, seed)
    if method == "kcenter":
        return select_kcenter(x, budget)
    if method == "density_kcenter":
        return select_weighted_kcenter(x, budget, np.maximum(density, 1e-6))
    if method == "density_balanced_kcenter":
        # Density-moderated: q is a density percentile. Idempotent for full_knn
        # density; converts anchor/min-max density to ranks first.
        q = density_percentile_ranks(density)
        return select_weighted_kcenter(x, budget, np.maximum(q * (1.0 - q), 1e-6))
    raise ValueError(f"Unsupported method: {method}")
