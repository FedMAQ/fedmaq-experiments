# ADR-0022 — Equal-weight, unfiltered server-side ensemble averaging is a constraint, not a claim of optimality

**Status**: Accepted · 2026-09-11
**Related**: ADR-0021 (Precision Scaling / Tier 2 power-mean family — the isolated independent variable this decision protects), ADR-0016 (FedMAQv2 server-KD repair study — a separate, later exploration touching adjacent territory)

## Context

FedMAQ's second aggregation stage — server-side, proxy-based, multi-teacher
ensemble distillation — combines client teacher predictions by unfiltered,
equal-weight averaging. Two papers curated in `fedmaq-wiki` as counterpoints
argue directly against this:

- **AE-KD** (du-2020) frames plain averaging as the `C = 1/M` endpoint of a
  weighted multi-objective formulation and argues for solving the weighting.
- **Selective-FD** (shao-2024) argues part of the ensemble should be dropped,
  showing that under one-class-per-client skew, two data-free baselines that
  consume the ensemble unfiltered collapse to the independent-learning floor.

No ADR previously recorded why FedMAQ freezes the simpler choice against these
direct objections. [fedmaq-literature#42](https://github.com/FedMAQ/fedmaq-literature/issues/42)
asks for that record, with three requirements: a practicality ground, a scope
ground, and an explicit acknowledgement that this is a constraint, not a claim
that equal-weight averaging is optimal — neither counterpoint paper is refuted
by this decision.

This ADR argues on design grounds only. Zero runs have completed under the
registered protocol; no outcome or results claim is made or implied.

## Decision

### Equal-weight averaging is a deliberate control, not an oversight

The thesis's studied independent variable is Precision Scaling — Tier 1 (hard
per-client memory clamp) and Tier 2 (soft quality signal, power-mean family,
[ADR-0021](0021-power-mean-formulation-family.md)). Weighted or filtered
ensemble aggregation was considered when the server-side distillation stage
was designed, and set aside intentionally: introducing a second,
independently-tunable degree of freedom into the aggregation step would
confound the isolation the Tier 1/Tier 2 study depends on. Equal-weight
averaging is the simplest aggregation rule that adds no such confound, which
is why it — rather than any weighted or filtered scheme — was the one
adopted.

This is the practicality ground: not an estimate of implementation or
client-side engineering cost, but a methodological cost. Weighting or
filtering the ensemble would reopen a second tuning surface (teacher weights,
or a drop threshold) at exactly the aggregation step whose output feeds the
signal this thesis studies, undermining the ability to attribute any observed
effect to Precision Scaling alone.

### Aggregation weighting is out of scope, not out of consideration

The scope ground follows directly: aggregation strategy is held as a control
across the experiment design
([experiment-design.md](../../.agents/rules/experiment-design.md)) so that
Precision Scaling remains the thesis's sole isolated independent variable.
This is not a judgment that aggregation weighting is unimportant, or that
AE-KD's or Selective-FD's findings don't apply to FedMAQ's setting — it is a
decision, made early, to aim the thesis's research question at signal
combination (how to best combine multiple per-client quality signals into one
scaling decision) rather than at ensemble combination (how to best combine
multiple teachers' predictions).

### Neither counterpoint is refuted

AE-KD's argument — that plain averaging is one endpoint of a more general
weighted formulation — is not contested here; FedMAQ's server-side stage sits
at that endpoint by choice, not because the general formulation was found
wanting. Selective-FD's finding — that unfiltered ensembles can collapse
toward independent learning under severe skew — is also not contested;
FedMAQ has not run the skew regime that finding was demonstrated under, and
this ADR makes no claim about how FedMAQ would behave there. Both papers'
proposals remain live candidates for a different, later research question
than this thesis asks.

### Relationship to the FedMAQv2 server-KD repair study

[ADR-0016](0016-v2-evaluation-protocol-and-advance-rule.md) separately
defines five repair candidate families for a v2 exploration, two of which —
`confidence_filter` and `disagreement_weight` — touch adjacent territory
(dropping or weighting teachers). That study is its own exploration design,
scoped against its own evidence, with its own open disposition; it does not
supersede or motivate this decision, and this ADR makes no claim about
whether or how those candidates will be pursued.

### Future work: a possible later isolation study, not a commitment

A natural follow-up, once the Tier 1/Tier 2 signal-combination formulation is
established, is a separate, later study that isolates the aggregation
mechanism itself — varying it independently to see whether it significantly
influences the chosen formulation's behavior. This would be a second,
deliberately distinct experiment, not a continuation or an implicit promise;
nothing in this ADR commits the thesis to running it.

## Consequences

- Every experiment run under the registered protocol uses unfiltered,
  equal-weight server-side ensemble averaging; no arm varies aggregation
  weighting.
- A future contributor proposing weighted or filtered aggregation for FedMAQ
  should treat it as a new, separate research question — requiring its own
  isolation from Precision Scaling — not as a bug fix or a completion of the
  current design.
- `fedmaq-literature`'s AE-KD and Selective-FD pages should link back to this
  ADR as the answer to the objections they raise; that link is
  literature-repo-scoped follow-up work, not made from this repo.
- If [ADR-0016](0016-v2-evaluation-protocol-and-advance-rule.md)'s
  `confidence_filter` or `disagreement_weight` candidates are ever advanced,
  that work should cross-reference this ADR rather than silently reopening
  the same confound this decision was written to avoid.
