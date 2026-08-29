from __future__ import annotations

from dataclasses import dataclass

import pytest
from omegaconf import OmegaConf


@dataclass
class _FakeTelemetry:
    events: list[str]

    def finish(self) -> None:
        self.events.append("telemetry.finish")


@dataclass
class _FakePersistence:
    events: list[str]

    def cleanup(self) -> None:
        self.events.append("persistence.cleanup")


def make_simulation_assembly():
    from fedmaq.simulation import SimulationAssembly

    return SimulationAssembly(
        client_app="client-app",
        server_app="server-app",
        num_supernodes=2,
        backend_config={"client_resources": {"num_cpus": 1}},
    )


def test_simulation_run_finalizes_resources_after_success():
    from fedmaq.simulation import SimulationRun

    events: list[str] = []
    telemetry = _FakeTelemetry(events)
    persistence = _FakePersistence(events)

    def runner(**kwargs):
        events.append("runner")
        assert kwargs == {
            "server_app": "server-app",
            "client_app": "client-app",
            "num_supernodes": 2,
            "backend_config": {"client_resources": {"num_cpus": 1}},
        }

    result = SimulationRun(
        assembly=make_simulation_assembly(),
        telemetry=telemetry,
        persistence_scope=persistence,
        runner=runner,
    ).execute()

    assert result is telemetry
    assert events == ["runner", "telemetry.finish", "persistence.cleanup"]


@pytest.mark.parametrize("error", [RuntimeError("simulation failed"), TimeoutError("timed out")])
def test_simulation_run_finalizes_resources_and_propagates_failure(error):
    from fedmaq.simulation import SimulationRun

    events: list[str] = []
    telemetry = _FakeTelemetry(events)
    persistence = _FakePersistence(events)

    def runner(**_kwargs):
        events.append("runner")
        raise error

    simulation = SimulationRun(
        assembly=make_simulation_assembly(),
        telemetry=telemetry,
        persistence_scope=persistence,
        runner=runner,
    )

    with pytest.raises(type(error), match=str(error)):
        simulation.execute()

    assert events == ["runner", "telemetry.finish", "persistence.cleanup"]


def test_simulation_run_cleans_persistence_if_telemetry_finalization_fails():
    from fedmaq.simulation import SimulationRun

    events: list[str] = []

    @dataclass
    class RaisingTelemetry:
        events: list[str]

        def finish(self) -> None:
            self.events.append("telemetry.finish")
            raise RuntimeError("telemetry flush failed")

    telemetry = RaisingTelemetry(events)
    persistence = _FakePersistence(events)

    with pytest.raises(RuntimeError, match="telemetry flush failed"):
        SimulationRun(
            assembly=make_simulation_assembly(),
            telemetry=telemetry,
            persistence_scope=persistence,
            runner=lambda **_kwargs: events.append("runner"),
        ).execute()

    assert events == ["runner", "telemetry.finish", "persistence.cleanup"]


def test_simulation_run_preserves_timeout_when_finalization_also_fails():
    from fedmaq.simulation import SimulationRun

    events: list[str] = []

    @dataclass
    class RaisingTelemetry:
        events: list[str]

        def finish(self) -> None:
            self.events.append("telemetry.finish")
            raise RuntimeError("telemetry flush failed")

    telemetry = RaisingTelemetry(events)
    persistence = _FakePersistence(events)

    def timed_out(**_kwargs):
        events.append("runner")
        raise TimeoutError("timed out")

    with pytest.raises(TimeoutError, match="timed out"):
        SimulationRun(
            assembly=make_simulation_assembly(),
            telemetry=telemetry,
            persistence_scope=persistence,
            runner=timed_out,
        ).execute()

    assert events == ["runner", "telemetry.finish", "persistence.cleanup"]


def test_builder_cleans_run_owned_persistence_if_app_construction_fails(tmp_path, monkeypatch):
    import fedmaq.simulation as simulation

    class FakeTelemetry:
        log_dir = tmp_path / "logs"

        def __init__(self, _config):
            self.finished = False

        def init_wandb(self) -> None:
            pass

        def finish(self) -> None:
            self.finished = True

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "algorithm": {"name": "fedkd"},
            "dataset": {"name": "mnist", "num_classes": 10},
            "heterogeneity": {"alpha": 0.1, "partition": "dirichlet"},
            "experiment": {
                "strict_determinism": False,
                "num_clients": 2,
                "num_public_samples": 1,
                "batch_size": 2,
                "client_gpus": 0.0,
                "client_fraction": 1.0,
                "total_rounds": 1,
            },
        }
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(simulation, "set_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        simulation,
        "generate_partition_indices",
        lambda **_kwargs: ([], {0: [], 1: []}),
    )
    monkeypatch.setattr(simulation, "TelemetryManager", FakeTelemetry)
    monkeypatch.setattr(simulation, "write_run_manifest", lambda *_args: None)

    def fail_app_construction(**_kwargs):
        raise RuntimeError("app construction failed")

    monkeypatch.setattr(simulation, "ClientApp", fail_app_construction)

    with pytest.raises(RuntimeError, match="app construction failed"):
        simulation.SimulationBuilder(cfg).build()

    persistence_root = tmp_path / ".data_partitions" / "fedkd_runs"
    assert persistence_root.exists()
    assert not list(persistence_root.iterdir())


def test_builder_keeps_persistence_driver_owned_when_callbacks_are_serialized(
    tmp_path, monkeypatch
):
    import ray.cloudpickle as cloudpickle

    import fedmaq.simulation as simulation

    class FakeTelemetry:
        log_dir = tmp_path / "logs"

        def __init__(self, _config):
            pass

        def init_wandb(self) -> None:
            pass

        def finish(self) -> None:
            pass

    class FakeApp:
        def __init__(self, **kwargs):
            self.callback = next(iter(kwargs.values()))

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "algorithm": {"name": "fedkd"},
            "dataset": {"name": "mnist", "num_classes": 10},
            "heterogeneity": {"alpha": 0.1, "partition": "dirichlet"},
            "experiment": {
                "strict_determinism": False,
                "num_clients": 2,
                "num_public_samples": 1,
                "batch_size": 2,
                "client_gpus": 0.0,
                "client_fraction": 1.0,
                "total_rounds": 1,
            },
        }
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(simulation, "set_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        simulation,
        "generate_partition_indices",
        lambda **_kwargs: ([], {0: [], 1: []}),
    )
    monkeypatch.setattr(simulation, "TelemetryManager", FakeTelemetry)
    monkeypatch.setattr(simulation, "write_run_manifest", lambda *_args: None)
    monkeypatch.setattr(simulation, "ClientApp", FakeApp)
    monkeypatch.setattr(simulation, "ServerApp", FakeApp)

    builder = simulation.SimulationBuilder(cfg)
    prepared = builder.build()

    assert builder.persistence_scope is None
    assert prepared.persistence_scope is not None
    cloudpickle.dumps(prepared.assembly.client_app.callback)
    cloudpickle.dumps(prepared.assembly.server_app.callback)
