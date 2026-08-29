from __future__ import annotations

import subprocess

from scripts.matrix_executor import MatrixExecutor
from scripts.matrix_planner import plan_matrix


def _matrix() -> dict:
    return {
        "phase": "smoke",
        "experiment_group": "planner_probe",
        "dataset": "cifar10",
        "model": "mobilenetv2",
        "total_rounds": 1,
        "client_gpus": 0.0,
        "seeds": [0, 1],
        "heterogeneities": ["dirichlet_alpha_0.1"],
        "runs": [
            {"alg": "fedavg", "label": "control"},
            {"alg": "fedprox", "label": "prox"},
        ],
    }


def test_matrix_plan_owns_order_filtering_and_sharding(tmp_path):
    plan = plan_matrix(
        tmp_path / "probe.yaml",
        _matrix(),
        overrides=["ray.object_store_gb=4"],
        shard=(2, 2),
    )

    assert [task.canonical_index for task in plan.canonical_tasks] == [1, 2, 3, 4]
    assert [task.canonical_index for task in plan.tasks] == [2, 4]
    assert [task.algorithm for task in plan.tasks] == ["fedprox", "fedprox"]
    assert "ray.object_store_gb=4" in plan.tasks[0].command
    assert plan.status_path.name == "sweep_status.shard-2-of-2.json"

    filtered = plan_matrix(tmp_path / "probe.yaml", _matrix(), only_labels=["prox"])
    assert [task.canonical_index for task in filtered.canonical_tasks] == [1, 2, 3, 4]
    assert [task.canonical_index for task in filtered.tasks] == [2, 4]
    assert {task.algorithm for task in filtered.tasks} == {"fedprox"}


def test_matrix_executor_is_testable_without_the_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = plan_matrix(tmp_path / "probe.yaml", _matrix())
    dispatched: list[tuple[list[str], dict]] = []
    cleanups: list[int] = []

    def fake_run(command, **kwargs):
        dispatched.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    executor = MatrixExecutor(
        plan,
        run=fake_run,
        cleanup=lambda: cleanups.append(1),
        sleep=lambda _seconds: None,
        clock=iter(range(8)).__next__,
        now=lambda: "2026-08-29T00:00:00",
        hostname="planner-host",
    )

    status = executor.execute()

    assert len(dispatched) == 4
    assert all(command[0][4] == "dataset=cifar10" for command in dispatched)
    assert status["state"] == "finished"
    assert status["completed"] == 4
    assert all(run["host"] == "planner-host" for run in status["runs"])
    assert len(cleanups) == 5
