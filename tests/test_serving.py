"""End-to-end test of the serving layer on a synthetic on-disk dataset.

Builds tiny feature stores + a COCO file + fake source images, registers them as a
backend dataset, and drives the FastAPI app through calibration and a couple of review
rounds -- submitting labels from the synthetic GT. Checks the batch/label/triage contract
and the crop endpoint. No real data or GPU needed.

The serving layer is orchestration only; this verifies the HTTP wiring, not the science
(the science is covered by tests/test_triage_session.py and regression/).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from protorcd_pipeline.triage.acquisition import l2_normalize


def _write_store(dirpath: Path, embeddings: np.ndarray, metas):
    dirpath.mkdir(parents=True, exist_ok=True)
    np.savez(dirpath / "chunk_0000.npz", embeddings=embeddings.astype(np.float32))
    (dirpath / "manifest.json").write_text(json.dumps({"chunk_files": ["chunk_0000.npz"]}))
    with (dirpath / "metadata.jsonl").open("w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")


def _build_synth_dataset(root: Path, n_per_class=30, n_fg=3, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_fg + 1, dim))
    xs, ys = [], []
    for cls in range(n_fg + 1):
        xs.append(centers[cls] + 0.12 * rng.normal(size=(n_per_class, dim)))
        ys.append(np.full(n_per_class, cls, dtype=np.int64))
    x = l2_normalize(np.vstack(xs).astype(np.float32))
    y = np.concatenate(ys)
    perm = rng.permutation(len(y))
    x, y = x[perm], y[perm]
    bg_x = l2_normalize(x + 0.02 * rng.normal(size=x.shape).astype(np.float32))
    fg_x = l2_normalize(x + 0.02 * rng.normal(size=x.shape).astype(np.float32))

    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    metas, coco_images, coco_anns = [], [], []
    for i in range(len(y)):
        fn = f"img_{i}.jpg"
        Image.new("RGB", (40, 40), color=(i % 255, 0, 0)).save(images_dir / fn)
        bbox = [5, 5, 20, 20]
        metas.append({"proposal_id": f"p{i}", "image_id": i, "file_name": fn, "bbox_xywh": bbox})
        coco_images.append({"id": i, "file_name": fn})
        if int(y[i]) != 0:
            coco_anns.append({"id": i, "image_id": i, "category_id": int(y[i]),
                              "bbox": bbox, "iscrowd": 0})
    _write_store(root / "bg", bg_x, metas)
    _write_store(root / "fg", fg_x, metas)
    coco = {"images": coco_images,
            "categories": [{"id": c, "name": f"sp{c}"} for c in range(1, n_fg + 1)],
            "annotations": coco_anns}
    (root / "coco.json").write_text(json.dumps(coco))
    return y


def test_serving_end_to_end(tmp_path):
    from fastapi.testclient import TestClient
    from serving.server import DatasetConfig, create_app

    root = tmp_path / "ds"
    _build_synth_dataset(root, n_per_class=30, n_fg=3, dim=32, seed=1)

    cfg = DatasetConfig(
        name="synth",
        bg_features=[str(root / "bg")],
        fg_features=[str(root / "fg")],
        coco=[str(root / "coco.json")],
        image_root=str(root / "images"),
        iou_threshold=0.25,
    )
    app = create_app({"synth": cfg})
    client = TestClient(app)

    assert client.get("/datasets").json()["datasets"] == ["synth"]

    # Create with offline calibration (GT auto-answers the calibration batch), small
    # batch/calibration so the synthetic pool supports a few rounds.
    r = client.post("/sessions", json={
        "dataset": "synth", "seed": 3, "batch_size": 10,
        "calibration_size": 20, "max_rounds": 3, "offline_calibration": True,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    sid = body["session_id"]
    assert body["state"]["calibrated"] is True
    assert body["state"]["rule"] is not None
    assert body["next_batch"] is not None
    assert body["next_batch"]["phase"] == "review"

    # Drive review rounds: answer every proposal as background=0 (valid labels), just to
    # exercise the contract deterministically.
    batch = body["next_batch"]
    assert batch["batch_id"]                       # batch carries a stable id
    seen_rounds = 0
    while batch is not None and seen_rounds < 3:
        # crop endpoint works for a proposal in the batch
        pid = batch["proposals"][0]["proposal_id"]
        cr = client.get(f"/crops/{sid}/{pid}")
        assert cr.status_code == 200 and cr.headers["content-type"] == "image/jpeg"
        assert len(cr.content) > 0

        labels = {p["proposal_id"]: 0 for p in batch["proposals"]}
        lr = client.post(f"/sessions/{sid}/labels",
                         json={"labels": labels, "batch_id": batch["batch_id"]})
        assert lr.status_code == 200, lr.text
        payload = lr.json()
        # triage payload present and internally consistent
        if payload["triage"] is not None:
            counts = payload["triage"]["counts"]
            assert set(counts) == {"resolve", "clear", "defer"}
            assert sum(counts.values()) == len(payload["triage"]["decisions"])
        # SAFEGUARD: re-submitting the SAME (now consumed) batch_id must be rejected 409,
        # not silently re-applied.
        dup = client.post(f"/sessions/{sid}/labels",
                          json={"labels": labels, "batch_id": batch["batch_id"]})
        assert dup.status_code == 409, dup.text
        seen_rounds += 1
        batch = payload["next_batch"]

    assert seen_rounds >= 1
    # state endpoint reflects progress and exposes explicit status
    st = client.get(f"/sessions/{sid}").json()
    assert st["cumulative_reviews"] >= 10
    assert st["status"] in {"awaiting_labels", "advancing", "finished"}
    assert st["failed"] is False
    assert client.delete(f"/sessions/{sid}").json()["deleted"] == sid


def test_stale_batch_id_rejected(tmp_path):
    """A submission with a wrong/old batch_id is rejected (409) without consuming labels."""
    from fastapi.testclient import TestClient
    from serving.server import DatasetConfig, create_app

    root = tmp_path / "ds"
    _build_synth_dataset(root, n_per_class=25, n_fg=3, dim=32, seed=2)
    cfg = DatasetConfig(
        name="synth", bg_features=[str(root / "bg")], fg_features=[str(root / "fg")],
        coco=[str(root / "coco.json")], image_root=str(root / "images"),
    )
    client = TestClient(create_app({"synth": cfg}))
    body = client.post("/sessions", json={
        "dataset": "synth", "seed": 1, "batch_size": 10,
        "calibration_size": 20, "max_rounds": 2, "offline_calibration": True,
    }).json()
    sid = body["session_id"]
    batch = body["next_batch"]
    labels = {p["proposal_id"]: 0 for p in batch["proposals"]}
    bad = client.post(f"/sessions/{sid}/labels",
                      json={"labels": labels, "batch_id": "not-the-current-batch"})
    assert bad.status_code == 409, bad.text
    # the real batch_id still works afterwards (labels weren't consumed)
    ok = client.post(f"/sessions/{sid}/labels",
                     json={"labels": labels, "batch_id": batch["batch_id"]})
    assert ok.status_code == 200, ok.text


def test_concurrent_double_submit_exactly_one_wins(tmp_path):
    """Two clients submitting the SAME batch_id simultaneously: exactly one succeeds, the
    other is rejected as stale. Guards the submit_labels() atomic critical section."""
    import threading

    from serving.session_manager import SessionManager, StaleBatchError
    from protorcd_pipeline.triage import load_pool, labels_from_coco

    root = tmp_path / "ds"
    y = _build_synth_dataset(root, n_per_class=30, n_fg=3, dim=32, seed=4)
    pool = load_pool(str(root / "bg"), str(root / "fg"))
    gt, _ = labels_from_coco(pool.metadata, str(root / "coco.json"), 0.25)
    pool.gt_labels = gt
    idx = pool.id_to_index

    mgr = SessionManager()
    m = mgr.create(pool, seed=2, batch_size=10, calibration_size=20, max_rounds=3,
                   image_root=str(root / "images"))
    # answer calibration to reach a review batch
    b = m.wait_for_batch()
    m.submit_labels({pid: int(gt[idx[pid]]) for pid in b.proposal_ids}, batch_id=b.batch_id)
    b = m.wait_for_batch()
    assert b.phase == "review"

    labels = {pid: int(gt[idx[pid]]) for pid in b.proposal_ids}
    batch_id = b.batch_id

    # Fire N submissions for the same batch at once, released together by a barrier to
    # maximize the chance of interleaving inside submit_labels().
    n = 8
    barrier = threading.Barrier(n)
    results: List[str] = []
    results_lock = threading.Lock()

    def worker():
        barrier.wait()
        try:
            m.submit_labels(labels, batch_id=batch_id, timeout=30.0)
            outcome = "ok"
        except StaleBatchError:
            outcome = "stale"
        except Exception as exc:  # any other error is a real failure of the test
            outcome = f"error:{type(exc).__name__}"
        with results_lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert results.count("ok") == 1, f"expected exactly one winner, got {results}"
    assert results.count("stale") == n - 1, f"expected {n - 1} stale rejections, got {results}"


def test_label_taxonomy_validation(tmp_path):
    """Out-of-taxonomy / negative label ids are rejected (400) and never reach the bank;
    valid ids pass. GET /datasets/{name}/labels exposes the taxonomy."""
    from fastapi.testclient import TestClient
    from serving.server import DatasetConfig, create_app

    root = tmp_path / "ds"
    _build_synth_dataset(root, n_per_class=25, n_fg=3, dim=32, seed=5)  # classes 1..3
    cfg = DatasetConfig(
        name="synth", bg_features=[str(root / "bg")], fg_features=[str(root / "fg")],
        coco=[str(root / "coco.json")], image_root=str(root / "images"),
    )
    client = TestClient(create_app({"synth": cfg}))

    # taxonomy endpoint derives {1,2,3} from COCO categories
    tax = client.get("/datasets/synth/labels").json()
    assert tax["validated"] is True
    assert set(tax["labels"].keys()) == {"1", "2", "3"}
    assert tax["background_id"] == 0

    body = client.post("/sessions", json={
        "dataset": "synth", "seed": 1, "batch_size": 10,
        "calibration_size": 20, "max_rounds": 2, "offline_calibration": True,
    }).json()
    sid = body["session_id"]
    assert set(body["labels"].keys()) == {"1", "2", "3"}   # taxonomy returned on create
    batch = body["next_batch"]
    ids = [p["proposal_id"] for p in batch["proposals"]]

    # an out-of-taxonomy positive id -> 400, batch NOT consumed
    bad = dict.fromkeys(ids, 0)
    bad[ids[0]] = 999999
    r = client.post(f"/sessions/{sid}/labels", json={"labels": bad, "batch_id": batch["batch_id"]})
    assert r.status_code == 400, r.text

    # a negative id -> 400
    neg = dict.fromkeys(ids, 0)
    neg[ids[0]] = -1
    r = client.post(f"/sessions/{sid}/labels", json={"labels": neg, "batch_id": batch["batch_id"]})
    assert r.status_code == 400, r.text

    # valid ids (0 and a real class) -> 200; the batch was still answerable after rejects
    good = dict.fromkeys(ids, 0)
    good[ids[0]] = 2
    r = client.post(f"/sessions/{sid}/labels", json={"labels": good, "batch_id": batch["batch_id"]})
    assert r.status_code == 200, r.text


def _tiny_pool(metadata):
    """A minimal ProposalPool carrying just the metadata CropService needs."""
    from protorcd_pipeline.triage import ProposalPool
    n = len(metadata)
    x = np.zeros((n, 2), dtype=np.float32)
    ids = [m["proposal_id"] for m in metadata]
    return ProposalPool(bg_x=x, fg_x=x, proposal_ids=ids, metadata=metadata)


def _make_image(path: Path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (30, 30), color=color).save(path)


def test_cropservice_single_root(tmp_path):
    """Single-root dataset resolves crops against image_root exactly as before."""
    from serving.crops import CropService
    root = tmp_path / "imgs"
    _make_image(root / "a.jpg", (10, 20, 30))
    pool = _tiny_pool([{"proposal_id": "p0", "file_name": "a.jpg", "bbox_xywh": [2, 2, 10, 10]}])
    svc = CropService(pool, image_root=root)
    assert len(svc.crop_bytes("p0")) > 0


def test_cropservice_multi_root_resolves_by_split(tmp_path):
    """Multi-root dataset resolves each proposal against its _review_split's root; the
    SAME relative file_name under different roots yields different images."""
    from serving.crops import CropService
    train_root = tmp_path / "train"
    test_root = tmp_path / "test"
    _make_image(train_root / "00/x.jpg", (255, 0, 0))    # red under train
    _make_image(test_root / "00/x.jpg", (0, 0, 255))     # blue under test
    pool = _tiny_pool([
        {"proposal_id": "t", "file_name": "00/x.jpg", "bbox_xywh": [1, 1, 20, 20], "_review_split": "train"},
        {"proposal_id": "e", "file_name": "00/x.jpg", "bbox_xywh": [1, 1, 20, 20], "_review_split": "test"},
    ])
    svc = CropService(pool, image_roots={"train": train_root, "test": test_root})
    train_bytes = svc.crop_bytes("t")
    test_bytes = svc.crop_bytes("e")
    assert len(train_bytes) > 0 and len(test_bytes) > 0
    # Same relative name, different roots -> genuinely different pixels (different bytes).
    assert train_bytes != test_bytes


def test_cropservice_unknown_split_fails_cleanly(tmp_path):
    """A proposal whose _review_split has no configured root fails explicitly (no silent
    fallback to some arbitrary root)."""
    from serving.crops import CropService
    train_root = tmp_path / "train"
    _make_image(train_root / "00/x.jpg", (1, 2, 3))
    pool = _tiny_pool([
        {"proposal_id": "u", "file_name": "00/x.jpg", "bbox_xywh": [1, 1, 5, 5], "_review_split": "val"},
    ])
    svc = CropService(pool, image_roots={"train": train_root})   # no "val", no default
    try:
        svc.crop_bytes("u")
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised


def test_cropservice_default_root_for_untagged(tmp_path):
    """With both image_roots and a default image_root: a tagged proposal uses its split
    root; an untagged proposal falls back to image_root -- explicit and deterministic."""
    from serving.crops import CropService
    default_root = tmp_path / "default"
    train_root = tmp_path / "train"
    _make_image(default_root / "d.jpg", (9, 9, 9))
    _make_image(train_root / "d.jpg", (200, 100, 50))
    pool = _tiny_pool([
        {"proposal_id": "tagged", "file_name": "d.jpg", "bbox_xywh": [1, 1, 10, 10], "_review_split": "train"},
        {"proposal_id": "untagged", "file_name": "d.jpg", "bbox_xywh": [1, 1, 10, 10]},  # no split
    ])
    svc = CropService(pool, image_root=default_root, image_roots={"train": train_root})
    tagged = svc.crop_bytes("tagged")     # -> train_root
    untagged = svc.crop_bytes("untagged")  # -> default_root
    assert len(tagged) > 0 and len(untagged) > 0
    assert tagged != untagged             # resolved to different trees


def test_cropservice_requires_a_root():
    """Constructing with neither image_root nor image_roots is rejected."""
    from serving.crops import CropService
    pool = _tiny_pool([{"proposal_id": "p", "file_name": "a.jpg", "bbox_xywh": [0, 0, 1, 1]}])
    try:
        CropService(pool)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_worker_failure_becomes_explicit(tmp_path, monkeypatch):
    """If the worker raises inside the session, the session transitions to FAILED and the
    error is surfaced -- the UI never blocks forever waiting for a batch."""
    from serving.session_manager import SessionManager, SessionStatus, SessionError
    from protorcd_pipeline.triage import load_pool, labels_from_coco
    import protorcd_pipeline.triage.session as session_mod

    root = tmp_path / "ds"
    y = _build_synth_dataset(root, n_per_class=20, n_fg=3, dim=32, seed=3)
    pool = load_pool(str(root / "bg"), str(root / "fg"))
    gt, _ = labels_from_coco(pool.metadata, str(root / "coco.json"), 0.25)
    pool.gt_labels = gt

    # Force the very first thing the worker does (build_calibration_bank) to blow up.
    def boom(self, *a, **k):
        raise RuntimeError("injected worker failure")
    monkeypatch.setattr(session_mod.TriageSession, "build_calibration_bank", boom)

    mgr = SessionManager()
    m = mgr.create(pool, seed=1, batch_size=5, calibration_size=10, image_root=str(root / "images"))
    # wait_for_batch must RAISE (SessionError), not hang, once the worker has failed.
    raised = False
    try:
        m.wait_for_batch(timeout=10.0)
    except SessionError:
        raised = True
    assert raised
    assert m.failed and m.status is SessionStatus.FAILED
    assert m.error is not None and "injected worker failure" in str(m.error)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_serving_end_to_end(Path(d))
    print("OK: serving end-to-end test passed")
