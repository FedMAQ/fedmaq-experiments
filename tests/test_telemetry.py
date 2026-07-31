"""Tests for the TelemetryManager CSV schema (Architecture Deepening Step 1).

These pin the contract the schema-stability comment in ``telemetry.py`` claims
but which had no test before this refactor: (1) a hook-declared metric key
reserves its CSV column even if absent from the first logged round, and (2) an
undeclared/unknown key never grows or duplicates the header after it is fixed.
"""

import csv
from pathlib import Path

from fedmaq.core.strategy_hooks import get_strategy_hook
from fedmaq.core.telemetry import TelemetryManager

CONF_DIR = str((Path(__file__).parent.parent / "conf").resolve())


def _make_manager(tmp_path, monkeypatch):
    monkeypatch.setattr("fedmaq.core.telemetry._HYDRA_AVAILABLE", False)
    tm = TelemetryManager({"experiment": {"telemetry": {"wandb_enabled": False}}})
    tm.log_dir = tmp_path
    tm.jsonl_path = tmp_path / "experiment_log.jsonl"
    tm.csv_path = tmp_path / "experiment_log.csv"
    return tm


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_hook_metric_key_reserves_header_column_before_it_appears(tmp_path, monkeypatch):
    """FedMAQ's grad-norm/q stats only exist starting round 1's fit, but the
    hook declares them via metric_keys() up front, so round 0's write must
    already reserve the column instead of silently dropping it later."""
    tm = _make_manager(tmp_path, monkeypatch)
    hook = get_strategy_hook("fedmaq", {"algorithm": {"name": "fedmaq"}})
    tm.register_hook_metric_keys(hook.metric_keys())

    tm.log(round_num=0, metrics={"round": 0, "test/accuracy": 0.1})
    tm.log(round_num=1, metrics={"round": 1, "test/accuracy": 0.2, "algorithm/fedmaq/avg_q": 6.0})

    rows = _read_csv(tm.csv_path)
    header = rows[0]
    assert "algorithm/fedmaq/avg_q" in header
    assert len(rows) == 3  # header + round 0 + round 1, no duplicate header
    assert rows[1][header.index("algorithm/fedmaq/avg_q")] == ""
    assert rows[2][header.index("algorithm/fedmaq/avg_q")] == "6.0"


def test_undeclared_key_is_appended_once_and_never_duplicates_header(tmp_path, monkeypatch):
    """An unexpected key not covered by any hook's metric_keys() is folded into
    the schema (sorted) on the first write, matching ``extrasaction='ignore'``
    -- it must never trigger a second header row or reorder later columns."""
    tm = _make_manager(tmp_path, monkeypatch)
    tm.log(round_num=0, metrics={"round": 0, "test/accuracy": 0.1, "debug/unexpected_key": 42})

    header_after_first_write = _read_csv(tm.csv_path)[0]
    assert "debug/unexpected_key" in header_after_first_write

    # A later round introduces yet another unknown key not present in round 0.
    tm.log(
        round_num=1,
        metrics={
            "round": 1,
            "test/accuracy": 0.2,
            "debug/unexpected_key": 99,
            "debug/never_seen_before": 7,
        },
    )

    rows = _read_csv(tm.csv_path)
    assert rows[0] == header_after_first_write  # schema locked after round 0
    assert len(rows) == 3  # still exactly one header + two data rows
    assert "debug/never_seen_before" not in rows[0]  # silently dropped, not appended


def test_a_wandb_failure_costs_visibility_and_nothing_else(tmp_path, monkeypatch):
    """WandB runs online for the grid, so its network is now a per-round dependency.

    Every reported result is computed from the local CSV, so a WandB error must
    neither propagate into the strategy's evaluate() (which would abort a run
    mid-sweep) nor pre-empt that round's local write. Before the ordering was
    reversed, ``self.run.log`` came first and unguarded, so a dropped connection
    at round 57 of 100 both killed the run and lost the row.
    """

    class ExplodingRun:
        def log(self, metrics, step=None):
            raise ConnectionError("simulated WandB outage")

        def finish(self):
            raise ConnectionError("simulated WandB outage at flush")

    tm = _make_manager(tmp_path, monkeypatch)
    tm.enabled = True
    tm.run = ExplodingRun()

    tm.log(round_num=0, metrics={"round": 0, "test/accuracy": 0.1})
    tm.log(round_num=1, metrics={"round": 1, "test/accuracy": 0.2})
    tm.finish()

    rows = _read_csv(tm.csv_path)
    assert len(rows) == 3, "both rounds must reach the CSV despite the WandB failures"
    header = rows[0]
    assert rows[1][header.index("test/accuracy")] == "0.1"
    assert rows[2][header.index("test/accuracy")] == "0.2"


def test_confirmatory_runs_log_online_while_smoke_runs_stay_off():
    """The grid is watchable remotely; ``experiment=ci`` must not need a key.

    ci.yaml composes from default.yaml, so it inherits telemetry unless it says
    otherwise -- without the override, a 2-round smoke test would try to reach
    WandB and would litter the project with runs once a key is present.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        for experiment, enabled, mode in (
            ("default", True, "online"),
            ("femnist", True, "online"),
            ("ci", False, "offline"),
            ("preliminary", False, "offline"),
        ):
            cfg = compose(config_name="config", overrides=[f"experiment={experiment}"])
            assert cfg.experiment.telemetry.wandb_enabled is enabled, experiment
            assert cfg.experiment.telemetry.mode == mode, experiment
