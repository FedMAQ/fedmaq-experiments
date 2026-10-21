# FedMAQ experiments

Code, frozen configuration, and run machinery behind the FedMAQ study: a
matched-budget evaluation of memory-bounded adaptive quantization and ensemble
distillation in federated learning. FedMAQ is compared with FedAvg, FedPAQ,
DAdaQuant, FedKD, and other baselines in simulated federated learning on
CIFAR-10, CIFAR-100, and FEMNIST, using one shared byte-accounting layer so
methods are compared at the same communication budget.

Companion repositories (all under the [FedMAQ](https://github.com/FedMAQ)
organization):

- `fedmaq-analyses`: scorers, tidy result tables, and figures computed from the
  run evidence.
- `fedmaq-manuscript`: the thesis text.
- `fedmaq-journal-article`: the journal article.

## Layout

| Path | Contents |
| --- | --- |
| `src/fedmaq/` | Simulation, baselines, quantization planner, byte accounting, evaluation |
| `conf/` | Hydra configuration for algorithms, datasets, heterogeneity, and run matrices |
| `scripts/` | Run, matrix-planning, freeze-check, and audit entry points |
| `tests/` | Unit and fixture tests |
| `docs/adr/` | Decision records for the method, protocol, and tooling |
| `docs/freeze/` | Frozen configuration, expected-run ledger, and gate evidence |

## Pre-registration

The evaluation protocol was frozen before the confirmatory runs. The git tags
`pre-registration`, `pre-registration-stage1a`, `pre-registration-fedkd-fix`,
and `pre-registration-manifest-fix` mark the frozen states. `docs/freeze/`
holds the machine-checked manifest, and `scripts/check_freeze.py --check`
verifies that the working tree still matches it.

## Setup and checks

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required. Training
runs need a CUDA GPU; the test suite does not.

```sh
uv sync
just check        # freeze, format, lint, typecheck, tests, generated artifacts, assurance
```

Without `just`, run the same steps with `uv run python -m pytest`,
`uv run python -m ruff check .`, and `uv run python scripts/check_freeze.py --check`.
How runs are dispatched and polled is described in
`docs/agents/execution-model.md`.

## Data

Datasets are downloaded by the loaders on first use and are not stored in this
repository. The full run evidence behind the reported results (per-run logs,
metrics, and sweep status files, with SHA-256 checksums) is archived on Zenodo.
DOI: to be added once the archive is published.

## License

The code is released under the [Apache License 2.0](LICENSE). The CIFAR-10,
CIFAR-100, and FEMNIST datasets keep their own licenses and terms.

## Citation

See [`CITATION.cff`](CITATION.cff), which GitHub renders as a "Cite this
repository" button. A Zenodo DOI will be added to it for the archived release.
