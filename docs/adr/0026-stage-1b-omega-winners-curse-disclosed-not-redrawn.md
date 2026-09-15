# ADR-0026 — Stage 1b's reused ω=0.5 cell is disclosed as winner's-curse biased, not redrawn

**Status**: Accepted · 2026-09-16
**Related**: [ADR-0021](0021-power-mean-formulation-family.md) (D2, amended below),
[ADR-0024](0024-stage-1a-seed-extension-and-stage-1b-omega-arm.md) (D2, resolved by this
ADR), [ADR-0012](0012-formulation-selection-and-the-iso-byte-amendment.md) (the iso-byte
tie rule this decision relies on)

## Context

ADR-0021 D2 fixes Stage 1b as reuse, not a fresh draw: "Sweep `p` at `ω=0.5`, then add
`ω ∈ {0.25, 0.75}` at the selected `p`, reusing the existing `ω=0.5` cell." That reused
cell is not an arbitrary observation of `ω=0.5`. It is the winner of a
maximum-over-seven-`p` comparison in Stage 1a, so plugging it into Stage 1b's three-way
`ω` comparison enters an already-selected optimum against two `ω` arms that have not been
selected for anything. `select_power_mean_omega_iso_byte` additionally breaks ties
neutral-first toward `ω=0.5` (`scripts/analysis.py:1987`), which was written when `ω=0.5`
was simply the shipped default, not a Stage-1a winner. Together, Stage 1b is structurally
primed to return `ω̂ = 0.5` independent of whether `ω=0.5` is actually best.

ADR-0024 D2 recorded this bias, established that the cheap repair — rerunning a fresh
`ω=0.5` arm at the same five seeds — reproduces bit-identically under this repo's
golden-repeatability gates and is therefore a no-op, and deferred the real fork: disclose
the bias in prose, or remove it by drawing all three `ω` arms at seeds outside
`{0, 7, 21, 42, 123}`. It required that fork to resolve in its own ADR before any Stage 1b
cell runs, because it amends ADR-0021 D2.

The author has chosen disclosure. The reason given is budget, not confidence that the bias
is small: Stage 1a already spent 168 cells (`power_mean_design`, ADR-0024 D1) reaching
`selected_p = 0.5`, and a fresh-seed redraw would need a materially larger, differently
shaped matrix than the twelve-cell `power_mean_omega` group registered at
`conf/protocol/replacement-v1.yaml:50`.

## Decision

### D1 — The fresh-seed redraw is declined

Registering all three `ω` arms at seeds outside `{0, 7, 21, 42, 123}` would remove the
bias by construction, but at a cost ADR-0024 already priced: it is not an extra rung on
the existing twelve-cell ladder, it is a new selection domain, since `replacement-v1.yaml`
pins `seeds: [0, 42, 123]` (extended to five only for `baseline_tuning_wide`) as the
registered base for every other stage. Declined on budget grounds, given the GPU time
already committed to Stage 1a.

### D2 — The winner's-curse bias is disclosed wherever the ω verdict is presented

Every document that presents Stage 1b's `ω` verdict states, alongside it:

- The `ω=0.5` cell entering Stage 1b is the Stage-1a `p`-sweep's maximum-over-seven
  winner, reused rather than freshly drawn, and reuse of a selected maximum is an
  upward-biased estimate of that cell's own performance.
- `select_power_mean_omega_iso_byte`'s tie rule breaks ties neutral-first toward `ω=0.5`
  (`scripts/analysis.py:1987`), so `ω̂ = 0.5` is the structurally expected outcome of
  Stage 1b independent of whether `ω=0.5` is actually the best weighting.
- Consequently, an `ω̂ = 0.5` result from Stage 1b is not evidence for `ω=0.5`. It is the
  absence of a rejection under a comparison structurally tilted toward that outcome — the
  same degeneracy-vs-significance distinction recorded in
  [`.agents/rules/experiment-design.md`](../../.agents/rules/experiment-design.md#paired-seed-selection-statistics).
  Only `ω̂ ∈ {0.25, 0.75}`, i.e. a margin large enough to overcome both the bias and the
  tie rule, would constitute evidence.

This amends ADR-0021 D2: the two-stage, reuse-not-redraw design stands, but the reused
cell's status as a Stage-1a winner — and the resulting one-sided prior toward `ω̂ = 0.5` —
must accompany every presentation of the Stage 1b result, not just this ADR.

## Consequences

- `conf/matrix/power_mean_omega.yaml`'s two `algorithm.p=???` placeholders may now be
  written in as `algorithm.p=0.5` (`selected_p`, ADR-0024 D1), and
  `matrix_contracts.power_mean_omega.sha256` at `conf/protocol/replacement-v1.yaml:50`
  updated to match, per the file's own re-registration procedure. This ADR authorizes the
  *disclosure decision*, not that write-in or the Stage 1b dispatch it unblocks; both
  remain separately gated by `AGENTS.md`'s frozen-config clause and require their own
  explicit authorization.
- `docs/research/2026-09-14-stage1a-power-mean-selection-analysis.md` §4.3 and §5 need a
  follow-up edit: D2 is resolved (disclosure, not redraw), so the document should stop
  describing it as the open blocking gate and instead point here.
- The disclosure text in D2 above must be carried into whatever document eventually
  presents the Stage 1b `ω` verdict (a Stage 1b analysis doc, and the manuscript's
  formulation-selection section), not restated ad hoc each time — link this ADR.
- No GPU cell runs as a result of this ADR by itself.
