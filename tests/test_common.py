"""Tests for scripts/common.py."""

import subprocess

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
