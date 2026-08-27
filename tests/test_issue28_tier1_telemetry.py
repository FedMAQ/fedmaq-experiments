import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

from fedmaq.core.quantization_planner import (
    QuantizationPlanner,
    QuantPlan,
    compute_fedmaq_q_k_t,
    compute_fedmaq_q_k_t_details,
)
from fedmaq.core.strategy_hooks.fedmaq import FedMAQHook
from fedmaq.core.telemetry import TelemetryManager

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from memory_ceiling import build_tier1_ceiling_frame, write_tier1_ceiling_analysis


def test_planner_details_report_raw_ceiling_soft_target_and_final_q():
    details = compute_fedmaq_q_k_t_details(
        c_k=4096.0,
        c_unit=512.0,
        g_k=1.0,
        g_max=1.0,
        n_k=100,
        n_max=100,
        formulation=0,
        q_min=1,
        q_max=16,
    )

    assert details.q_k_max == pytest.approx(8.0)
    assert details.q_hat == pytest.approx(16.0)
    assert details.q == 8
    assert compute_fedmaq_q_k_t(
        c_k=4096.0,
        c_unit=512.0,
        g_k=1.0,
        g_max=1.0,
        n_k=100,
        n_max=100,
        formulation=0,
        q_min=1,
        q_max=16,
    ) == details.q


def test_planner_propagates_ceiling_details_without_changing_assignments(monkeypatch):
    planner = QuantizationPlanner("fedmaq", lambda _dataset, _classes: None)
    monkeypatch.setattr(planner, "_ensure_grad_norm_model", lambda _parameters, _ctx: None)
    monkeypatch.setattr(
        planner,
        "_probe_grad_norms",
        lambda *_args: ([1.0, 1.0], [100, 100]),
    )

    plan = planner.plan_round(
        parameters=None,
        client_pids=[0, 1],
        client_cids=["0", "1"],
        client_indices_dict={},
        client_memory=[4096.0, 16384.0],
        ctx=None,
        qp_cfg={
            "q_min": 1,
            "q_max": 16,
            "c_unit": 512.0,
            "formulation": 0,
            "resource_aware": True,
        },
        seed_base=42,
        server_round=1,
    )

    assert plan.client_q == {"0": 8, "1": 16}
    assert plan.client_q_max == {"0": 8.0, "1": 32.0}
    assert plan.client_q_hat == {"0": 16.0, "1": 16.0}
    assert plan.tier1_enabled is True


def test_hook_aggregates_ceiling_distribution_and_binding_fraction():
    hook = FedMAQHook({"algorithm": {"name": "fedmaq"}})
    hook._current_plan = QuantPlan(
        client_q={"0": 2, "1": 8, "2": 4},
        grad_norms=[],
        client_q_max={"0": 2.0, "1": 8.0, "2": 16.0},
        client_q_hat={"0": 4.0, "1": 8.0, "2": 4.0},
        tier1_enabled=True,
    )

    metrics = hook.get_eval_metrics(None, 1)

    assert metrics["algorithm/fedmaq/min_q_k_max"] == pytest.approx(2.0)
    assert metrics["algorithm/fedmaq/avg_q_k_max"] == pytest.approx(26 / 3)
    assert metrics["algorithm/fedmaq/max_q_k_max"] == pytest.approx(16.0)
    expected_std = pd.Series([2, 8, 16]).std(ddof=0)
    assert metrics["algorithm/fedmaq/std_q_k_max"] == pytest.approx(expected_std)
    assert metrics["algorithm/fedmaq/tier1_binding_fraction"] == pytest.approx(1 / 3)


def test_non_resource_aware_plan_does_not_emit_tier1_metrics():
    hook = FedMAQHook({"algorithm": {"name": "fedmaq"}})
    hook._current_plan = QuantPlan(
        client_q={"0": 8},
        grad_norms=[],
        client_q_max={"0": 2.0},
        client_q_hat={"0": 16.0},
        tier1_enabled=False,
    )

    metrics = hook.get_eval_metrics(None, 1)

    assert not any("q_k_max" in key or "tier1_binding" in key for key in metrics)


def test_uniform_memory_control_reports_constant_ceiling_and_zero_binding():
    hook = FedMAQHook({"algorithm": {"name": "fedmaq"}})
    hook._current_plan = QuantPlan(
        client_q={"0": 4, "1": 8, "2": 16},
        grad_norms=[],
        client_q_max={"0": 16.0, "1": 16.0, "2": 16.0},
        client_q_hat={"0": 4.0, "1": 8.0, "2": 16.0},
        tier1_enabled=True,
    )

    metrics = hook.get_eval_metrics(None, 1)

    assert metrics["algorithm/fedmaq/avg_q_k_max"] == pytest.approx(16.0)
    assert metrics["algorithm/fedmaq/std_q_k_max"] == pytest.approx(0.0)
    assert metrics["algorithm/fedmaq/tier1_binding_fraction"] == pytest.approx(0.0)


def test_tier1_metrics_are_serialized_in_the_stable_csv_schema(tmp_path):
    telemetry = TelemetryManager({"experiment": {"telemetry": {"wandb_enabled": False}}})
    telemetry.jsonl_path = tmp_path / "experiment_log.jsonl"
    telemetry.csv_path = tmp_path / "experiment_log.csv"
    telemetry.register_hook_metric_keys(FedMAQHook({"algorithm": {"name": "fedmaq"}}).metric_keys())
    telemetry.log(
        1,
        {
            "round": 1,
            "algorithm/fedmaq/avg_q_k_max": 8.0,
            "algorithm/fedmaq/tier1_binding_fraction": 0.5,
        },
    )

    with telemetry.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["algorithm/fedmaq/avg_q_k_max"] == "8.0"
    assert rows[0]["algorithm/fedmaq/tier1_binding_fraction"] == "0.5"
    assert rows[0]["algorithm/fedmaq/max_q_k_max"] == ""


def test_memory_ceiling_analysis_returns_plot_ready_long_frame(tmp_path):
    first = (
        tmp_path
        / "outputs"
        / "formal"
        / "cifar10_mobilenetv2"
        / "benchmark_grid"
        / "fedmaq"
        / "dirichlet_alpha_0.1"
        / "seed_0"
        / "experiment_log.csv"
    )
    second = (
        tmp_path
        / "outputs"
        / "formal"
        / "cifar10_mobilenetv2"
        / "uniform_memory_control"
        / "fedmaq"
        / "dirichlet_alpha_0.1"
        / "seed_0"
        / "experiment_log.csv"
    )
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "round": [1, 2],
            "algorithm/fedmaq/avg_q": [4.0, 5.0],
            "algorithm/fedmaq/avg_q_k_max": [3.0, 6.0],
            "algorithm/fedmaq/tier1_binding_fraction": [1.0, 0.0],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "round": [1, 2],
            "algorithm/fedmaq/avg_q": [8.0, 8.0],
            "algorithm/fedmaq/avg_q_k_max": [16.0, 16.0],
            "algorithm/fedmaq/tier1_binding_fraction": [0.0, 0.0],
        }
    ).to_csv(second, index=False)

    frame = build_tier1_ceiling_frame([first, second])

    assert set(frame["run"]) == {
        "formal/cifar10_mobilenetv2/benchmark_grid/fedmaq/dirichlet_alpha_0.1/seed_0",
        "formal/cifar10_mobilenetv2/uniform_memory_control/fedmaq/dirichlet_alpha_0.1/seed_0",
    }
    assert list(frame.columns) == [
        "run",
        "round",
        "realized_q",
        "tier1_ceiling_q",
        "tier1_binding_fraction",
    ]
    uniform_run = (
        "formal/cifar10_mobilenetv2/uniform_memory_control/fedmaq/"
        "dirichlet_alpha_0.1/seed_0"
    )
    assert frame.loc[frame["run"] == uniform_run, "tier1_ceiling_q"].tolist() == [16.0, 16.0]

    output_prefix = tmp_path / "analysis" / "tier1_memory_ceiling"
    write_tier1_ceiling_analysis([first, second], output_prefix)
    assert output_prefix.with_suffix(".csv").is_file()
    assert output_prefix.with_suffix(".png").is_file()


def test_memory_ceiling_analysis_skips_non_resource_aware_logs(tmp_path):
    csv_path = tmp_path / "fedmaq_no_resource.csv"
    pd.DataFrame({"round": [1], "algorithm/fedmaq/avg_q": [8.0]}).to_csv(
        csv_path, index=False
    )

    assert build_tier1_ceiling_frame([csv_path]).empty
