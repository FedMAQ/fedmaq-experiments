"""Pytest configuration and shared fixtures for fedmaq tests."""

import pytest

from fedmaq.core.partitioning import _load_dataset_cached


@pytest.fixture(autouse=True)
def clear_dataset_cache():
    """Clear the dataset LRU cache before each test.

    Prevents stale mock datasets or real dataset objects from leaking between
    test cases that monkeypatch ``load_dataset``.
    """
    _load_dataset_cached.cache_clear()
    yield
    _load_dataset_cached.cache_clear()
