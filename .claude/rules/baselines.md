# Baseline Algorithm Stack

Implement under `src/fedmaq/baselines/` and track in `.claude/project/baseline_registry.md`:

| Group             | Algorithms        |
| ----------------- | ----------------- |
| Seminal Controls  | FedAvg, FedProx   |
| Pure Quantization | FedPAQ, DAdaQuant |
| Pure KD           | FedDistill (FedMD dropped — see `docs/DECISIONS.md` Decision 25) |
| Hybrid Q+KD       | FedKD (CFD dropped — see `docs/DECISIONS.md` Decision 26) |

Update `baseline_registry.md` when adding or porting a baseline.

## FedMD excluded from smoke/regression sweeps

FedMD is dropped from the formal grid (Decision 25) and is unlikely to reappear in
future experiments. It is also the slowest baseline by far (disk-persisted,
up to 4x `run_epochs` per round). Do not include it in smoke-test matrices
(`conf/matrix/*.yaml`) or `scripts/golden_diff.py`'s default `GOLDEN_SET` — see
Decision 45. Its code is retained for reproducibility; re-include it in a sweep
only when a change actually touches its code path.

## Client Model Persistence

For baselines where the server does not aggregate model weights (e.g., FedMD and other prediction-averaging / distillation baselines), client model state dicts are persisted on disk under:

- **Directory**: `.data_partitions/fedmd_models/` (located inside the gitignored partition cache)
- **Filename**: `client_{cid}.pth`
  This keeps client model states unified and prevents local weights from being lost across simulated rounds in Flower.
