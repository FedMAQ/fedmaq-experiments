"""Canonical run-tree fixtures shared by analysis and identity tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from fedmaq.core.manifest import build_manifest
from fedmaq.core.run_identity import get_canonical_output_dir
from scripts.analysis import RunRecord

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CanonicalRunTree:
    """Write the same artifacts and identity-bearing path as a real run."""

    root: Path

    def write_run(
        self,
        algorithm: str,
        formulation: int | str | None,
        seed: int,
        metrics: pd.DataFrame,
        *,
        group: str | None = None,
        alpha: float = 0.5,
        dataset: str = "cifar10",
        model: str = "mobilenetv2",
        algorithm_config: str | None = None,
        variant: str = "",
        refinements: tuple[bool, bool, bool] = (False, False, False),
        post_process: bool = False,
        phase: str | None = None,
        output_dir: Path | None = None,
    ) -> RunRecord:
        """Create a canonical run directory and return its analysis record.

        ``metrics`` is the only test-specific input. The helper owns the path,
        resolved config, Hydra choice record, CSV artifact, and provenance
        manifest so tests cannot accidentally describe a flat or incompatible
        run tree.
        """
        group = group or ("formulation_study" if algorithm == "fedmaq" else "benchmark_grid")
        algorithm_config = algorithm_config or algorithm
        phase = phase or ("formal" if group in {"benchmark_grid", "ablation"} else "explore")
        if not variant and group == "formulation_study" and algorithm_config == "fedmaq":
            variant = f"f{formulation}"

        relative_dir = output_dir or get_canonical_output_dir(
            phase=phase,
            dataset=dataset,
            model=model,
            exp_group=group,
            algorithm=algorithm_config,
            heterogeneity=f"dirichlet_alpha_{alpha}",
            seed=seed,
            variant=variant,
        )
        job_dir = self.root / relative_dir
        job_dir.mkdir(parents=True, exist_ok=True)
        hydra_dir = job_dir / ".hydra"
        hydra_dir.mkdir(exist_ok=True)

        config = {
            "_algorithm_config_name": algorithm_config,
            "dataset": {"name": dataset},
            "heterogeneity": {"alpha": alpha},
            "seed": seed,
            "algorithm": {
                "name": algorithm,
                "formulation": formulation,
                "soft_voting": refinements[0],
                "ema_student": refinements[1],
                "grad_norm_ema": refinements[2],
                "post_process": post_process,
            },
            "experiment": {"total_rounds": int(metrics["round"].max())},
        }
        OmegaConf.save(OmegaConf.create(config), hydra_dir / "config.yaml")
        OmegaConf.save(
            OmegaConf.create({"hydra": {"runtime": {"choices": {"algorithm": algorithm_config}}}}),
            hydra_dir / "hydra.yaml",
        )
        csv_path = job_dir / "experiment_log.csv"
        metrics.to_csv(csv_path, index=False)
        manifest = build_manifest(config, repo_root=REPO_ROOT)
        manifest["source_root"] = job_dir.resolve().as_posix()
        (job_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )

        return RunRecord(
            job_dir=job_dir,
            dataset=dataset,
            alpha=alpha,
            algorithm=algorithm,
            formulation=formulation,
            seed=seed,
            csv_path=csv_path,
            refinements=refinements,
            algorithm_config=algorithm_config,
            experiment_group=group,
            phase=phase,
            post_process=post_process,
            variant=variant,
            promotable=True,
        )


def write_run(
    root: Path,
    algorithm: str,
    formulation: int | str | None,
    seed: int,
    accs: list[float],
    mbs: list[float],
    *,
    rounds: list[int] | None = None,
    group: str | None = None,
    alpha: float = 0.5,
    dataset: str = "cifar10",
    variant: str = "",
    algorithm_config: str | None = None,
    refinements: tuple[bool, bool, bool] = (False, False, False),
    post_process: bool = False,
    phase: str | None = None,
    output_dir: Path | None = None,
) -> RunRecord:
    """Convenience adapter for the common round/accuracy/MB fixture shape."""
    metrics = pd.DataFrame(
        {
            "round": rounds or list(range(1, len(accs) + 1)),
            "test/accuracy": accs,
            "communication/cumulative_mb": mbs,
        }
    )
    return CanonicalRunTree(root).write_run(
        algorithm,
        formulation,
        seed,
        metrics,
        group=group,
        alpha=alpha,
        dataset=dataset,
        variant=variant,
        algorithm_config=algorithm_config,
        refinements=refinements,
        post_process=post_process,
        phase=phase,
        output_dir=output_dir,
    )


def run_fedmaq_q_transition_fixture(output_dir: Path) -> dict[str, object]:
    """Exercise a real FedMAQ post-process q transition and persist its record."""
    from flwr.app import RecordDict

    from fedmaq.baselines.postprocess import FedMAQPostProcessCompressionHook
    from fedmaq.core.wire_codec import unpack_quantized_tensor

    state = RecordDict()
    records: list[dict[str, object]] = []
    for round_number, q in enumerate((8, 8, 4), start=1):
        hook = FedMAQPostProcessCompressionHook(
            q=q,
            state=state,
            rng=np.random.default_rng(100 + round_number),
        )
        _, report = hook.compress([np.array([0.25, -0.5, 0.75], dtype=np.float32)])
        wire = unpack_quantized_tensor(report.payloads[0])
        records.append(
            {
                "round": round_number,
                "q": q,
                "is_diff": wire.is_diff,
                "bit_width": wire.bit_width,
                "payload_bytes": report.payload_bytes,
                "measured_bytes": report.measured_bytes,
            }
        )

    if [record["q"] for record in records] != [8, 8, 4]:
        raise AssertionError("fixture did not execute the requested q schedule")
    if not any(record["is_diff"] is True for record in records) or records[-1]["is_diff"]:
        raise AssertionError("fixture did not record a real q transition")

    fixture = {
        "schema_version": 1,
        "fixture": "fedmaq_q_transition",
        "ledger": "assurance",
        "records": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fedmaq_q_transition.json").write_text(
        json.dumps(fixture, indent=2) + "\n", encoding="utf-8"
    )
    return fixture
