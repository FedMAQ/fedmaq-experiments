"""Pytest configuration and shared fixtures for fedmaq tests."""

import pytest

from fedmaq.core.partitioning import _load_dataset_cached
from fedmaq.core.telemetry import TelemetryManager


@pytest.fixture(autouse=True)
def clear_dataset_cache():
    """Clear the dataset LRU cache before each test.

    Prevents stale mock datasets or real dataset objects from leaking between
    test cases that monkeypatch ``load_dataset``.
    """
    _load_dataset_cached.cache_clear()
    yield
    _load_dataset_cached.cache_clear()


@pytest.fixture(autouse=True)
def redirect_telemetry_logs(tmp_path, monkeypatch):
    """Redirect telemetry logs to a temporary directory to keep workspace root clean."""
    original_init = TelemetryManager.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.log_dir = tmp_path
        self.jsonl_path = tmp_path / "experiment_log.jsonl"
        self.csv_path = tmp_path / "experiment_log.csv"

    monkeypatch.setattr(TelemetryManager, "__init__", patched_init)
