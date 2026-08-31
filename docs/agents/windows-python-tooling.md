# Windows Python tooling

Use module execution for Python tools so the command resolves through the active interpreter instead of a generated Windows console launcher:

```powershell
uv run python -m pytest
uv run python -m ruff check .
uv run python -m <project_module> <subcommand>
```

For example, `fedmaq-literature` uses `uv run python -m fedmaq_literature.cli audit-coverage`.

When a command reports `uv trampoline failed to canonicalize script path`, repair the specific package from the repository root and retry. The error identifies a stale launcher or invalid embedded interpreter path:

```powershell
uv sync --reinstall-package pytest
```

Substitute the package that owns the failing command when it is not `pytest`. For a non-mutating diagnostic or when UV itself is unavailable, run the repository interpreter directly:

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check .
```

Treat `.venv` as generated state rather than source evidence. Prefer targeted launcher regeneration; it keeps the repair fast and avoids unnecessary scientific-package changes.
