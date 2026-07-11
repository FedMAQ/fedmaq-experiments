# Workspace Map

Five-repo thesis workspace. `fedmaq-experiments` is the domain "hub" — sibling
repos index its `.claude/rules/` rather than duplicating domain content.

| Repo                                              | Role                                                              | Agent entry                                     | Domain rules                       |
| -------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------ | ----------------------------------- |
| [fedmaq-experiments](../../)                       | FedMAQ code, Hydra, Flower, WandB                                  | [CLAUDE.md](../../CLAUDE.md)                     | **Owner:** `.claude/rules/`         |
| [fedmaq-literature](../../../fedmaq-literature/)   | PDFs, markdown conversions, OKF knowledge graph (`kg/`)            | [CLAUDE.md](../../../fedmaq-literature/CLAUDE.md) | `thesis-context.md` -> experiments  |
| [fedmaq-analyses](../../../fedmaq-analyses/)       | Notebooks, thesis figures                                          | [AGENTS.md](../../../fedmaq-analyses/AGENTS.md)  | `thesis-context.mdc` -> experiments |
| [fedmaq-manuscript](../../../fedmaq-manuscript/)   | LaTeX thesis (Ch 1-6 drafted, de-overclaim pass applied)           | [README.md](../../../fedmaq-manuscript/README.md) | **Owner:** `.claude/rules/`         |
| [fedmaq-presentations](../../../fedmaq-presentations/) | Beamer slides                                                  | [AGENTS.md](../../../fedmaq-presentations/AGENTS.md) | `thesis-context.mdc` -> experiments |

**Cross-repo rule:** Non-experiments repos must not duplicate domain content.
Index via `../fedmaq-experiments/.claude/rules/`.

**Agent tooling layout (mixed state):** `fedmaq-experiments`, `fedmaq-manuscript`,
and `fedmaq-literature` have migrated to Claude Code — each owns `.claude/rules/`,
`.claude/project/` (experiments also has `.claude/skills/`; literature's
`thesis-context.md` still points at a stale `fedmaq-experiments/.cursor/rules/`
path). `fedmaq-analyses` and `fedmaq-presentations` have not migrated yet and
still use `.cursor/rules/`, `.cursor/skills/`, `.cursor/project/`, with
`thesis-context.mdc` pointers to `fedmaq-experiments/.cursor/rules/` that are
likewise stale until each repo's own migration pass.
