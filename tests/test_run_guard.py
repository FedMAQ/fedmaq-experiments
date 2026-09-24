"""Concurrent-run isolation: sweep, run-directory, and host Ray locks, and notify_run.sh."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import run_guard  # noqa: E402
from scripts.run_guard import (  # noqa: E402
    LockHeldError,
    PriorEvidenceError,
    exclusive_lock,
    run_dir_guard,
)

_HOLDER = (
    "import sys, time\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from pathlib import Path\n"
    "from scripts.run_guard import exclusive_lock\n"
    "with exclusive_lock(Path(sys.argv[2]), 'holder'):\n"
    "    print('held', flush=True)\n"
    "    time.sleep(120)\n"
)


@pytest.fixture
def hold_lock():
    """Hold a lockfile from another process, as a live sweep would."""

    holders: list[subprocess.Popen[str]] = []

    def start(path: Path) -> subprocess.Popen[str]:
        proc = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, str(REPO_ROOT), str(path)],
            stdout=subprocess.PIPE,
            text=True,
        )
        holders.append(proc)
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "held"
        return proc

    yield start
    for proc in holders:
        proc.kill()
        proc.wait(timeout=30)


@pytest.fixture(autouse=True)
def isolated_lock_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(run_guard.LOCK_DIR_ENV, str(tmp_path / "host-locks"))


def _probe_sweep(tmp_path: Path, monkeypatch, experiment_group: str = "lock_probe") -> Path:
    matrix_dir = tmp_path / "conf" / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    (matrix_dir / f"{experiment_group}.yaml").write_text(
        "phase: smoke\n"
        f"experiment_group: {experiment_group}\n"
        "dataset: cifar10\n"
        "model: mobilenetv2\n"
        "total_rounds: 1\n"
        "seeds: [0]\n"
        "heterogeneities: [dirichlet_alpha_0.1]\n"
        "runs:\n"
        "  - alg: fedavg\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path / "outputs" / "smoke" / "cifar10_mobilenetv2" / experiment_group


def _planned_cells(tmp_path: Path, experiment_group: str = "lock_probe") -> list[Path]:
    from omegaconf import OmegaConf

    from scripts.matrix_planner import plan_matrix

    matrix = tmp_path / "conf" / "matrix" / f"{experiment_group}.yaml"
    return [task.output_dir for task in plan_matrix(matrix, OmegaConf.load(matrix)).tasks]


def _drive_sweep(monkeypatch, experiment_group: str = "lock_probe") -> dict[str, list]:
    import scripts.run_matrix as run_matrix

    effects: dict[str, list] = {"cleanups": [], "dispatched": []}
    monkeypatch.setattr(run_matrix, "kill_ray_processes", lambda: effects["cleanups"].append(1))
    monkeypatch.setattr(run_matrix.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        run_matrix.subprocess,
        "run",
        lambda cmd, *a, **k: (
            effects["dispatched"].append(cmd) or subprocess.CompletedProcess(cmd, 0)
        ),
    )
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--matrix", experiment_group])
    run_matrix.main()
    return effects


def test_a_second_sweep_on_a_live_group_refuses_and_kills_nothing(tmp_path, monkeypatch, hold_lock):
    group_dir = _probe_sweep(tmp_path, monkeypatch)
    first = hold_lock(group_dir / "sweep_status.lock")

    effects: dict[str, list] = {}
    with pytest.raises(SystemExit) as refusal:
        effects = _drive_sweep(monkeypatch)

    assert "sweep output group" in str(refusal.value.code)
    assert effects == {}
    assert not (group_dir / "sweep_status.json").exists(), "the live sweep's status is untouched"
    assert first.poll() is None, "the live sweep is still running"


def test_a_sweep_refuses_while_another_sweep_owns_this_hosts_ray(tmp_path, monkeypatch, hold_lock):
    _probe_sweep(tmp_path, monkeypatch)
    hold_lock(run_guard.ray_host_lock_path())

    effects: dict[str, list] = {}
    with pytest.raises(SystemExit) as refusal:
        effects = _drive_sweep(monkeypatch)

    assert "Ray cleanup on this host" in str(refusal.value.code)
    assert effects == {}


def test_a_sweep_refuses_a_cell_with_an_earlier_attempts_records(tmp_path, monkeypatch):
    import scripts.run_matrix as run_matrix

    group_dir = _probe_sweep(tmp_path, monkeypatch)
    cleanups: list[int] = []
    monkeypatch.setattr(run_matrix, "kill_ray_processes", lambda: cleanups.append(1))
    cell_dirs = [path for path in _planned_cells(tmp_path)]
    cell_dirs[0].mkdir(parents=True)
    (cell_dirs[0] / "v2_diagnostic.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--matrix", "lock_probe"])

    with pytest.raises(SystemExit) as refusal:
        run_matrix.main()

    assert "earlier attempt" in str(refusal.value.code)
    assert str(cell_dirs[0]) in str(refusal.value.code)
    assert cleanups == []
    assert not (group_dir / "sweep_status.json").exists()


def test_a_sweep_runs_and_releases_its_locks(tmp_path, monkeypatch):
    group_dir = _probe_sweep(tmp_path, monkeypatch)

    effects = _drive_sweep(monkeypatch)

    assert len(effects["dispatched"]) == 1 and effects["cleanups"]
    with exclusive_lock(group_dir / "sweep_status.lock", "next sweep"):
        with exclusive_lock(run_guard.ray_host_lock_path(), "next sweep"):
            pass


def test_a_crashed_holders_lock_is_recovered(tmp_path, hold_lock):
    lock = tmp_path / "group" / "sweep_status.lock"
    holder = hold_lock(lock)
    with pytest.raises(LockHeldError, match=r"is held by pid \d+ on host .* \(holder, since"):
        with exclusive_lock(lock, "contender"):
            pass

    holder.kill()
    holder.wait(timeout=30)
    assert lock.exists(), "the crashed holder leaves its lockfile behind"

    with exclusive_lock(lock, "successor"):
        assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == os.getpid()


@pytest.mark.parametrize("name", ["experiment_log.jsonl", "v2_diagnostic.jsonl"])
def test_a_run_refuses_to_append_to_an_earlier_attempts_records(tmp_path, name):
    earlier = '{"round": 1, "created_utc": "2026-09-20T00:00:00+00:00"}\n'
    (tmp_path / name).write_text(earlier, encoding="utf-8")

    with pytest.raises(PriorEvidenceError, match=name):
        with run_dir_guard(tmp_path):
            pytest.fail("the run must not start")

    assert (tmp_path / name).read_text(encoding="utf-8") == earlier


def test_a_run_directory_admits_one_writer(tmp_path, hold_lock):
    hold_lock(tmp_path / run_guard.RUN_LOCK_FILENAME)

    with pytest.raises(LockHeldError, match="run directory"):
        with run_dir_guard(tmp_path):
            pytest.fail("the run must not start")


def test_run_py_refuses_before_the_simulation_starts(tmp_path):
    earlier = "{}\n"
    (tmp_path / "v2_diagnostic.jsonl").write_text(earlier, encoding="utf-8")
    (tmp_path / ".hydra").mkdir()
    (tmp_path / ".hydra" / "config.yaml").write_text("earlier: attempt\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "scripts/run.py", f"hydra.run.dir={tmp_path.as_posix()}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode != 0
    assert "earlier attempt" in result.stderr
    assert not (tmp_path / "run_manifest.json").exists()
    assert (tmp_path / "v2_diagnostic.jsonl").read_text(encoding="utf-8") == earlier
    # Hydra would rewrite these and open run.log before the task function runs.
    assert (tmp_path / ".hydra" / "config.yaml").read_text(encoding="utf-8") == "earlier: attempt\n"
    assert not (tmp_path / ".hydra" / "overrides.yaml").exists()
    assert not (tmp_path / "run.log").exists()


_STUBBED_RUN_PY = (
    "import runpy, sys\n"
    "from pathlib import Path\n"
    "import fedmaq.simulation\n"
    "from hydra.core.hydra_config import HydraConfig\n"
    "def stub(cfg):\n"
    "    out = Path(HydraConfig.get().runtime.output_dir)\n"
    "    (out / 'stub_ran.txt').write_text(str(out), encoding='utf-8')\n"
    "fedmaq.simulation.run = stub\n"
    "sys.argv = ['scripts/run.py', *sys.argv[1:]]\n"
    "runpy.run_path('scripts/run.py', run_name='__main__')\n"
)


def test_run_py_runs_in_the_directory_it_guarded(tmp_path):
    run_dir = tmp_path / "cell"

    result = subprocess.run(
        [sys.executable, "-c", _STUBBED_RUN_PY, f"hydra.run.dir={run_dir.as_posix()}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    assert Path((run_dir / "stub_ran.txt").read_text(encoding="utf-8")) == run_dir.resolve()
    assert (run_dir / ".hydra" / "config.yaml").is_file()
    assert (run_dir / run_guard.RUN_LOCK_FILENAME).is_file()


def test_dry_run_flags_a_cell_that_would_refuse(tmp_path, monkeypatch, capsys):
    import scripts.run_matrix as run_matrix

    group_dir = _probe_sweep(tmp_path, monkeypatch)
    (cell_dir,) = _planned_cells(tmp_path)
    cell_dir.mkdir(parents=True)
    (cell_dir / "experiment_log.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--matrix", "lock_probe", "--dry_run"])

    run_matrix.main()

    assert "WILL REFUSE" in capsys.readouterr().out
    assert not (group_dir / "sweep_status.lock").exists(), "a dry run takes no lock"


def _posix_bash() -> str | None:
    bash = shutil.which("bash")
    if bash is None:
        return None
    lowered = bash.lower()
    if sys.platform == "win32" and ("system32" in lowered or "windowsapps" in lowered):
        return None
    return bash


def test_notify_run_returns_with_the_wrapped_command(tmp_path):
    """A heartbeat ``sleep`` that outlives the run holds the caller's pipe open."""

    bash = _posix_bash()
    if bash is None:
        pytest.skip("needs a POSIX bash, not the WSL launcher")
    (tmp_path / "scripts").mkdir()
    shutil.copy(REPO_ROOT / "scripts" / "notify_run.sh", tmp_path / "scripts")
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    curl = stub_bin / "curl"
    curl.write_text("#!/usr/bin/env bash\nprintf 200\n", encoding="utf-8", newline="\n")
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    env = {
        **os.environ,
        "NTFY_TOPIC": "stub-topic",
        "NTFY_TOKEN": "stub-token",
        "NTFY_SERVER": "http://127.0.0.1:9",
        "NTFY_HEARTBEAT_SEC": "30",
    }

    started = time.monotonic()
    result = subprocess.run(
        [bash, "-c", 'PATH="$(cd bin && pwd):$PATH" bash scripts/notify_run.sh true | cat'],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=25,
    )

    assert result.returncode == 0, result.stderr
    assert time.monotonic() - started < 10
