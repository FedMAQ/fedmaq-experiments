# Environment Variables

| Variable                   | Used by               | Notes                                         |
| -------------------------- | --------------------- | --------------------------------------------- |
| `FEDMAQ_QA_MIN_MEAN_GRADE` | literature            | Default `good` (Docling mean_grade threshold) |
| `FEDMAQ_QA_MIN_LOW_GRADE`  | literature            | Default `fair`                                |
| `FEDMAQ_MARKER_DEVICE`     | literature            | Override Marker device (`cuda` / `cpu`)       |
| `HF_HUB_DISABLE_SYMLINKS`  | literature            | Set automatically on Windows in Docling path  |
| `WANDB_API_KEY`            | experiments, analyses | Experiment tracking                           |

The literature RAG variables (`OPENROUTER_API_KEY`, `FEDMAQ_EMBED_*`) were
retired with the vector-RAG stack in the OKF restructure; only the
conversion-QA and Marker vars above remain.

Create `.env` locally (gitignored); document new vars here when added.

**Setup:** `uv sync` in each Python repo; `uv sync --extra dev` for pytest in
experiments.
