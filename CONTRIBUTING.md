# Contributing

Thank you for helping improve ProtoRCD Pipeline. Contributions may include bug reports, documentation improvements, tests, examples, serving-layer changes, and UI integrations built against the HTTP contract.

## Before contributing

1. Read the [README](README.md) — especially *Frozen provenance tags* and *Contribution rules* — and [docs/architecture.md](docs/architecture.md), which explains the layering rules.
2. Set up a development environment:

   ```bash
   pip install -e ".[serving,dev]"        # triage + HTTP service + pytest/ruff
   pip install -e ".[detect,features]"    # only if you are working on stages 1–2 (GPU / heavy ML stack)
   ```

3. Review open issues and pull requests to avoid duplicate work.
4. Do not submit credentials, private keys, proprietary data, restricted datasets, camera-trap imagery or coordinates that reveal sensitive locations (e.g. of protected species), personally identifiable information, or material you are not authorized to share.
5. Open an issue to report a problem or propose a change before beginning a substantial contribution (see [docs/MAINTAINER_ROLES.md](docs/MAINTAINER_ROLES.md) for the decision process).

## What can change, and where

The repository is split into layers with different rules. Knowing which one your change touches determines what is accepted:

| Area | Status | What is accepted |
|---|---|---|
| [`protorcd_pipeline/`](protorcd_pipeline/README.md) | **Frozen** at `scientific-parity-baseline` (`08e0fa0`) | No changes without explicit maintainer sign-off. Its algorithms reproduce the paper bit-for-bit and are guarded by the equivalence regression. Propose changes in an issue first. |
| [`serving/`](serving/README.md) | Active | Orchestration, IO, validation, and hardening around the frozen core. Must not implement scientific logic. Changes to request/response shapes must update [`serving/API.md`](serving/API.md). |
| UI code (external) | Active | Builds **only** against [`serving/API.md`](serving/API.md). Collects labels and renders decisions; never computes species/background decisions, calibration, acquisition, or retirement. |
| [`examples/`](examples/README.md), [`docs/`](docs/), [`tests/`](tests/) | Active | Improvements welcome. |
| [`regression/`](regression/README.md) | Guarded | Must keep validating against the frozen research runs; do not loosen checks to make a change pass. |

**Dependency direction is one-way:** `serving/` may import `protorcd_pipeline/`; `protorcd_pipeline/` must never import `serving/`.

**Provenance tags are immutable.** Never move or delete an existing tag; a newly validated stage gets a new tag, created by the release manager.

## Contribution pathway

The project welcomes contributions in increasing order of technical and maintenance responsibility:

1. Run an example (e.g. [`examples/run_triage_offline.py`](examples/run_triage_offline.py)) and report a problem.
2. Improve documentation or examples.
3. Add or improve a test.
4. Improve the serving layer or build a UI integration against the HTTP contract.
5. Propose a change to the scientific pipeline — only via an issue with a clear justification, since it requires sign-off and re-validation against the research runs.

## Testing

Any change that touches the backend must keep **both** suites green before it lands. If it can't, it is out of scope until discussed.

1. **Lint and format.** Configuration lives in [`pyproject.toml`](pyproject.toml):

   ```bash
   ruff check .
   ruff format --check .
   ```

2. **Unit/integration tests** (fast, no GPU or datasets needed):

   ```bash
   pytest tests/
   # fresh clone without installing: PYTHONPATH=. pytest tests/
   ```

   Add or update tests in [`tests/`](tests/) for any change in behaviour, particularly serving-layer validation and error paths.

3. **Research-equivalence regression** (heavier; requires the frozen research runs, feature stores, and annotations):

   ```bash
   regression/run_all.sh    # must end with: ALL REGRESSIONS PASSED (10/10)
   ```

   See [regression/README.md](regression/README.md) for what the checks cover and the density caveat. If you do not have access to the research data, say so in your pull request and a maintainer will run the sweep before merging.

## Pull requests

A pull request should:

- Reference the related issue or explain the problem being addressed.
- Be limited to one coherent change.
- State which layers it touches, and confirm `protorcd_pipeline/` is unchanged (or link the sign-off if it is not).
- Report the results of `pytest tests/` and, where applicable, `regression/run_all.sh`.
- Include or update tests when practical.
- Update documentation when user-visible behaviour, the HTTP contract, configuration, installation, or limitations change.
- Identify new dependencies (and which optional extra they belong to — stage 3 must stay free of torch), data assumptions, security implications, and maintenance implications.
- Not include secrets, model weights, feature stores, datasets, large unreviewed binary assets, or unlicensed materials.

Maintainers may request changes, defer a contribution, or decline it when the change lacks a clear maintenance owner, conflicts with project scope or the frozen-pipeline rules, introduces unacceptable security or data risks, or cannot be reviewed with available resources.

## License and contributor rights

By submitting a contribution, you represent that you have the right to submit it and that it may be distributed under this repository's [license](LICENSE). If your employer, institution, funder, or data provider imposes restrictions, obtain authorization before contributing.

## Security issues

Do not report suspected vulnerabilities in a public issue. Follow [SECURITY.md](SECURITY.md).
