# Baseline Algorithm Stack

Implement under `src/fedmaq/baselines/` and track in `.claude/project/baseline_registry.md`:

| Group             | Algorithms        |
| ----------------- | ----------------- |
| Seminal Controls  | FedAvg, FedProx   |
| Pure Quantization | FedPAQ, DAdaQuant |
| Pure KD           | FedMD, FedDistill |
| Hybrid Q+KD       | FedKD, CFD        |

Update `baseline_registry.md` when adding or porting a baseline.

## Client Model Persistence

For baselines where the server does not aggregate model weights (e.g., FedMD and other prediction-averaging / distillation baselines), client model state dicts are persisted on disk under:

- **Directory**: `.data_partitions/fedmd_models/` (located inside the gitignored partition cache)
- **Filename**: `client_{cid}.pth`
  This keeps client model states unified and prevents local weights from being lost across simulated rounds in Flower.
