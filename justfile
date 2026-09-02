set windows-shell := ["pwsh", "-NoProfile", "-Command"]

# Default recipe: run every local CI gate
default: check

# Run all standard verification checks
check: freeze format lint typecheck test generated assurance

freeze:
    uv run python scripts/check_freeze.py --check

format:
    uv run python -m ruff format --check .

# Run pytest test suite
test *args="":
    uv run python -m pytest {{args}}

# Run linter
lint:
    uv run python -m ruff check .

# Fix formatting and auto-fixable lint violations
fix:
    uv run python -m ruff format .
    -uv run python -m ruff check --fix .

# Run static type checking
typecheck:
    uv run python -m mypy

generated:
    uv run python scripts/dump_frozen_configs.py --check
    uv run python scripts/dump_expected_runs.py --check
    uv run python scripts/generate_schema_fixture.py --check
    uv run python scripts/stage_manifest.py --check
    uv run python scripts/stage_ledgers.py --check

assurance:
    uv run python scripts/run_assurance_fixtures.py
