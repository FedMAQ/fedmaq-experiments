# Workspace Map

Five-repo thesis workspace. `fedmaq-experiments` is the domain "hub" — sibling
repos index its `.claude/rules/` rather than duplicating domain content.

| Repo                                                   | Role                                                    | Agent entry                                          | Domain rules                       |
| ------------------------------------------------------ | ------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------- |
| [fedmaq-experiments](../../)                           | FedMAQ code, Hydra, Flower, WandB                       | [CLAUDE.md](../../CLAUDE.md)                         | **Owner:** `.claude/rules/`        |
| [fedmaq-literature](../../../fedmaq-literature/)       | PDFs, markdown conversions, OKF knowledge graph (`kg/`) | [CLAUDE.md](../../../fedmaq-literature/CLAUDE.md)    | `thesis-context.md` -> experiments |
| [fedmaq-analyses](../../../fedmaq-analyses/)           | Notebooks, thesis figures                               | [CLAUDE.md](../../../fedmaq-analyses/CLAUDE.md)      | `thesis-context.md` -> experiments |
| [fedmaq-manuscript](../../../fedmaq-manuscript/)       | LaTeX thesis (Ch 1-6 drafted)                           | [README.md](../../../fedmaq-manuscript/README.md)    | **Owner:** `.claude/rules/`        |
| [fedmaq-presentations](../../../fedmaq-presentations/) | Beamer slides                                           | [CLAUDE.md](../../../fedmaq-presentations/CLAUDE.md) | `thesis-context.md` -> experiments |

**Cross-repo rule:** Non-experiments repos must not duplicate domain content.
Index via `../fedmaq-experiments/.claude/rules/`.
