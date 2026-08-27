"""Tests for scripts/common.py."""

import subprocess

import pytest

from scripts import common


def test_kill_ray_processes_force_kills_on_linux(monkeypatch):
    """A wedged raylet/GCS survives ``ray stop`` alone (2026-08-01 pass2_factorial
    cascade: a timeout-killed run's Ray grandchildren outlived the driver and
    blocked the next task's ``ray.init()``). Non-Windows must force-kill them,
    mirroring the unconditional ``taskkill`` already done for Windows.
    """
    monkeypatch.setattr(common.sys, "platform", "linux")
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(common.subprocess, "run", fake_run)

    common.kill_ray_processes()

    assert calls[0] == ["uv", "run", "ray", "stop"]
    pkill_targets = {cmd[-1] for cmd in calls[1:]}
    assert pkill_targets == {"raylet", "gcs_server", "plasma_store"}
    for cmd in calls[1:]:
        assert cmd[:3] == ["pkill", "-9", "-f"]


def test_kill_ray_processes_uses_taskkill_on_windows(monkeypatch):
    monkeypatch.setattr(common.sys, "platform", "win32")
    monkeypatch.setattr(common.time, "sleep", lambda _seconds: None)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(common.subprocess, "run", fake_run)

    common.kill_ray_processes()

    assert calls[0] == ["uv", "run", "ray", "stop"]
    assert any(cmd[:2] == ["taskkill", "/F"] for cmd in calls[1:])
    assert not any(cmd[0] == "pkill" for cmd in calls[1:])


@pytest.mark.parametrize("count", [1, 2, 3, 4, 7])
def test_partition_tasks_is_disjoint_and_complete(count):
    tasks = [{"canonical_index": index} for index in range(1, 18)]
    shards = [
        {task["canonical_index"] for task in common.partition_tasks(tasks, index, count)}
        for index in range(1, count + 1)
    ]

    assert set().union(*shards) == set(range(1, 18))
    assert sum(map(len, shards)) == len(tasks)
    for index, left in enumerate(shards):
        for right in shards[index + 1 :]:
            assert left.isdisjoint(right)


@pytest.mark.parametrize("value", ["0/3", "1/0", "4/3", "one/3", "1 / 3"])
def test_parse_shard_rejects_invalid_selectors(value):
    with pytest.raises(ValueError):
        common.parse_shard(value)


def test_parse_shard_is_one_based():
    assert common.parse_shard("2/5") == (2, 5)
    assert common.sharded_sweep_status_filename(2, 5) == "sweep_status.shard-2-of-5.json"


def test_validate_unique_output_dirs_rejects_collisions():
    with pytest.raises(ValueError, match="maps runs"):
        common.validate_unique_output_dirs(
            [
                {"label": "first", "output_dir": "outputs/same"},
                {"label": "second", "output_dir": "outputs/same"},
            ]
        )
