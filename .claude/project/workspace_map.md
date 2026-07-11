# Workspace Map

Five-repo thesis workspace. `fedmaq-experiments` is the domain "hub" — sibling
repos index its `.claude/rules/` rather than duplicating domain content.

| Repo                                              | Role                                                              | Agent entry                                     | Domain rules                       |
| -------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------ | ----------------------------------- |
| [fedmaq-experiments](../../)                       | FedMAQ code, Hydra, Flower, WandB                                  | [CLAUDE.md](../../CLAUDE.md)                     | **Owner:** `.claude/rules/`         |
| [fedmaq-literature](../../../fedmaq-literature/)   | PDFs, markdown conversions, OKF knowledge graph (`kg/`)            | [CLAUDE.md](../../../fedmaq-literature/CLAUDE.md) | `thesis-context.md` -> experiments  |
| [fedmaq-analyses](../../../fedmaq-analyses/)       | Notebooks, thesis figures                                          | [CLAUDE.md](../../../fedmaq-analyses/CLAUDE.md)  | `thesis-context.md` -> experiments |
| [fedmaq-manuscript](../../../fedmaq-manuscript/)   | LaTeX thesis (Ch 1-6 drafted, de-overclaim pass applied)           | [README.md](../../../fedmaq-manuscript/README.md) | **Owner:** `.claude/rules/`         |
| [fedmaq-presentations](../../../fedmaq-presentations/) | Beamer slides                                                  | [CLAUDE.md](../../../fedmaq-presentations/CLAUDE.md) | `thesis-context.md` -> experiments |

**Cross-repo rule:** Non-experiments repos must not duplicate domain content.
Index via `../fedmaq-experiments/.claude/rules/`.

**Locked cross-repo architectural decisions:**

| Topic              | Decision                                                                                                                                                                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Thesis context     | `fedmaq-experiments/.claude/rules/` is canonical (decomposed from a prior `context.md`)                                                                                                                                                      |
| Experiments layout | uv monorepo, code under `src/fedmaq/core/` and `src/fedmaq/baselines/`                                                                                                                                                                       |
| Tooling            | Preferred stack in `tech-stack.md`; adopt extra libs (pandas, sklearn, etc.) when justified                                                                                                                                                  |
| Literature PDFs    | Never parse `papers/*.pdf` in chat; pipeline + `markdown/` only                                                                                                                                                                              |
| Literature KG      | OKF bundle at `kg/` (see `fedmaq-literature/SPEC.md`); two layers — raw `markdown/` (citable) + curated OKF nodes. No vector store (grep + read); nodes authored directly, no approve gate (review via `git diff`); no cross-repo auto-edits |
| Analyses inputs    | WandB exports + Hydra outputs from experiments                                                                                                                                                                                               |
