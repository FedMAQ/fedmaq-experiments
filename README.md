# FedMAQ experiments

Research code, frozen configurations, and run machinery for a thesis study of memory-bounded adaptive quantization and ensemble distillation in simulated federated learning. FedMAQ and the baselines share communication-byte accounting. The first study's configuration and the subsequent registered protocols are recorded in versioned configuration.

The companion [analyses repository](https://github.com/FedMAQ/fedmaq-analyses) scores the archived runs; the [thesis manuscript](https://github.com/FedMAQ/fedmaq-manuscript) reports the findings.

## Quick start

Requires Python 3.11 or newer, [uv](https://docs.astral.sh/uv/), and a host with CUDA resources for the configured client allocation.

```sh
uv sync
uv run python scripts/run.py experiment=ci
```

This runs a 2-round, 2-client CIFAR-10 FedAvg smoke check with Weights & Biases telemetry disabled. The run requests `client_gpus: 1.0` per simulated client and writes to `outputs/<date>/<time>/` by default. Use the registered matrices for study reproduction.

Formal and exploratory sweeps use declared matrix files through `uv run python scripts/run_matrix.py --matrix <name>`. See the [experiment guide](docs/experiments/README.md), [execution model](docs/agents/execution-model.md), and the registered matrix contracts in `conf/protocol/` before choosing a matrix. Do not treat every file in `conf/matrix/` as a current confirmatory run.

## Data and evidence

The loaders fetch CIFAR-10, CIFAR-100, MNIST, and Fashion-MNIST on demand through torchvision. FEMNIST uses the LEAF writer-partitioned dataset; when its files are absent, the loader fetches and unpacks the LEAF archive, then caches prepared arrays under `data/femnist_leaf/`. Generated partition indices are cached under `.data_partitions/`. These inputs and caches are not included in the Git checkout.

Matrix runs use `outputs/<phase>/<dataset_model>/<experiment_group>/<algorithm__variant>/<heterogeneity>/seed_<seed>/`. The checkout does not contain a complete versioned run-evidence archive or a linked Zenodo record. Reproducing the study's existing analysis requires supplied per-run metric logs (`experiment_log.csv`), Hydra configuration snapshots (`.hydra/config.yaml`), and `run_manifest.json` files under those paths. Sweep completeness audits also use `sweep_status.json`. A fresh run creates new evidence; it does not recreate the original study evidence.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/fedmaq/` | Flower simulation, methods, partitioning, evaluation, and byte accounting |
| `conf/` | Dataset, algorithm, experiment, matrix, and protocol configuration |
| `scripts/` | Single-run and matrix runners, analysis, and freeze/evidence checks |
| `docs/adr/`, `docs/freeze/` | Design decisions, frozen configuration, and protocol evidence |
| `tests/` | Unit and fixture tests |

Run `just check` for freeze validation, formatting, lint, type checking, tests, generated-artifact checks, and assurance fixtures. The steps are defined in [`justfile`](justfile).

## Citation, license, and help

Cite the software using [`CITATION.cff`](CITATION.cff). The project code is licensed under the [Apache License 2.0](LICENSE); dataset licenses and terms remain separate. For project questions or issues, use the [GitHub issue tracker](https://github.com/FedMAQ/fedmaq-experiments/issues).
