"""Exclusive ownership of sweep groups, run directories, and the host's Ray processes.

Each lock is an OS lock on a file held open for the holder's lifetime:
``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows. The kernel drops the
lock when the holder exits, crashed or not, so a lockfile left behind by a dead
process is stale by construction and the next acquirer simply takes it. The file
also records the holder's PID, host, and purpose, but only to explain a refusal;
it is never consulted to decide ownership. Lockfiles are never unlinked on release,
because a new process could otherwise lock a fresh inode at the same path while an
older holder still owns the unlinked one.

``flock`` rather than ``lockf``: POSIX record locks belong to the process, so a
second acquire from the same process would silently succeed.
"""

from __future__ import annotations

import getpass
import json
import os
import socket
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

RUN_LOCK_FILENAME = ".fedmaq-run.lock"
LOCK_DIR_ENV = "FEDMAQ_LOCK_DIR"
PRIOR_EVIDENCE_FILENAMES = ("experiment_log.jsonl", "v2_diagnostic.jsonl")

# Windows byte-range locks are mandatory, so the locked byte sits far past the
# holder metadata that a refused process still needs to read.
_WINDOWS_LOCK_OFFSET = 1 << 30


class LockHeldError(RuntimeError):
    """Another live process owns the resource."""


class PriorEvidenceError(RuntimeError):
    """A run directory already holds records from an earlier attempt."""


def _try_lock(stream: IO[str]) -> bool:
    if sys.platform == "win32":
        import msvcrt

        stream.seek(_WINDOWS_LOCK_OFFSET)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(stream: IO[str]) -> None:
    if sys.platform == "win32":
        import msvcrt

        stream.seek(_WINDOWS_LOCK_OFFSET)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _holder_description(path: Path) -> str:
    try:
        holder = json.loads(path.read_text(encoding="utf-8").strip("\0 \n"))
        return (
            f"pid {holder.get('pid')} on host {holder.get('host')} "
            f"({holder.get('purpose')}, since {holder.get('acquired_utc')})"
        )
    except (OSError, ValueError, AttributeError):
        return "an unidentified process"


@contextmanager
def exclusive_lock(path: Path, purpose: str) -> Iterator[None]:
    """Hold ``path`` exclusively, or raise :class:`LockHeldError` without waiting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        if not _try_lock(stream):
            raise LockHeldError(
                f"{purpose} is held by {_holder_description(path)}; lockfile {path}. "
                "Refusing to start: another live run owns it, and nothing was cleaned up."
            )
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "host": socket.gethostname(),
                        "purpose": purpose,
                        "acquired_utc": datetime.now(UTC).isoformat(),
                    }
                )
            )
            stream.flush()
            yield
        finally:
            # close() below releases the lock anyway; an unlock error must not
            # mask the body's exception.
            with suppress(OSError):
                _unlock(stream)
    finally:
        stream.close()


def ray_host_lock_path() -> Path:
    """Per-user host lock guarding :func:`scripts.common.kill_ray_processes`.

    Ray cleanup is ``ray stop`` plus a force-kill of every raylet, GCS, and plasma
    process the user can signal. Neither can be scoped to one ``ray.temp_dir``, so
    two sweeps with different temp dirs would still kill each other's actors. The
    lock is therefore host-wide per user rather than per temp dir.
    """

    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = str(os.getpid())
    root = Path(os.environ.get(LOCK_DIR_ENV) or tempfile.gettempdir())
    return root / f"fedmaq-ray-{user}.lock"


@contextmanager
def ray_cleanup_lock(purpose: str) -> Iterator[None]:
    with exclusive_lock(ray_host_lock_path(), f"Ray cleanup on this host ({purpose})"):
        yield


@contextmanager
def sweep_locks(status_path: Path) -> Iterator[None]:
    """Own one sweep's status file, then the host's Ray processes.

    The group lock sits beside the status file it protects, so each shard owns
    only its own status; the per-run lock catches two sweeps reaching one cell.
    """

    group_lock = status_path.with_name(f"{status_path.stem}.lock")
    with exclusive_lock(group_lock, f"sweep output group {status_path.parent}"):
        with ray_cleanup_lock(f"sweep {status_path}"):
            yield


def prior_evidence(run_dir: Path) -> list[Path]:
    """Return record files an earlier attempt left in ``run_dir``."""

    return [
        path
        for name in PRIOR_EVIDENCE_FILENAMES
        if (path := run_dir / name).is_file() and path.stat().st_size > 0
    ]


@contextmanager
def run_dir_guard(run_dir: Path) -> Iterator[None]:
    """Own ``run_dir`` and refuse to append to an earlier attempt's records.

    A run is not resumable mid-flight: it always starts at round 1, so records
    already present belong to another attempt and would interleave with this one.
    Recovery moves the failed attempt aside before rerunning the cell.
    """

    with exclusive_lock(run_dir / RUN_LOCK_FILENAME, f"run directory {run_dir}"):
        stale = prior_evidence(run_dir)
        if stale:
            names = ", ".join(path.name for path in stale)
            raise PriorEvidenceError(
                f"{run_dir} already holds records from an earlier attempt ({names}). "
                "Refusing to append. Preserve and move that attempt aside, then rerun."
            )
        yield
