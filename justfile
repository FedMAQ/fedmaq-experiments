set windows-shell := ["pwsh", "-NoProfile", "-Command"]

# Default recipe: run full test, lint, and type checks
default: check

# Run all standard verification checks
check: freeze lint test

freeze:
    uv run python scripts/check_freeze.py --check

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
