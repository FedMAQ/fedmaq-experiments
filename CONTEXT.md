# FedMAQ Thesis Domain

Multi-adaptive quantization and knowledge distillation for memory-constrained federated learning under non-IID data. Canonical glossary for terms shared across `fedmaq-experiments` (code) and `fedmaq-manuscript` (thesis) — resolves naming drift between the two.

> **This file is deliberately a glossary**, which the workspace's reference layout
> ([`../fedmaq-journal-paper/docs/adr/0012-agentic-context-layout.md`](../fedmaq-journal-paper/docs/adr/0012-agentic-context-layout.md))
> forbids for a `CONTEXT.md`. The exemption is explicit there: that repo avoids
> being a glossary *because* "All shared vocabulary defers to
> `fedmaq-experiments/CONTEXT.md`." This is the file that rule was written to
> protect. Do not "fix" it toward pointer-only. See
> [ADR-0014](docs/adr/0014-agentic-context-layout.md).

## Authority map

Six-repo thesis workspace. `fedmaq-experiments` is the domain hub — sibling repos
index its `.claude/rules/` rather than duplicating domain content.

| Repo | Owns | Agent entry |
| --- | --- | --- |
| [fedmaq-experiments](./) | Code, Hydra, Flower, WandB, **the domain rules** | `CLAUDE.md` |
| [fedmaq-literature](../fedmaq-literature/) | PDFs, markdown conversions, OKF knowledge graph | `CLAUDE.md` |
| [fedmaq-analyses](../fedmaq-analyses/) | Notebooks, thesis figures | `CLAUDE.md` |
| [fedmaq-manuscript](../fedmaq-manuscript/) | LaTeX thesis (Ch 1–6), **its own writing rules** | `README.md` |
| [fedmaq-presentations](../fedmaq-presentations/) | Beamer slides | `CLAUDE.md` |
| [fedmaq-journal-paper](../fedmaq-journal-paper/) | IEEE Access submission, **the agentic-context reference layout** ([ADR-0012](../fedmaq-journal-paper/docs/adr/0012-agentic-context-layout.md)) | `CLAUDE.md` |

**Cross-repo rule:** non-experiments repos must not duplicate domain content; they
index `../fedmaq-experiments/.claude/rules/`.

**Within this repo**, when two sources disagree: `conf/**` beats prose describing
it; `docs/adr/` beats everything for *why*; the pinned GitHub Issues beat every
file for *what is true right now*. Run counts, dispatch state and sync status live
only in Issues — if you find a number in a tracked file, it is stale by
construction.

## Working conventions

- **One canonical home per fact.** A number, status or decision lives in exactly
  one place; everything else points at it and never restates it.
- **No archives.** Git history is the record — a superseded doc is deleted, not
  parked in an `archive/` folder.
- **Reference lives behind pointers.** Settled, rarely-touched material belongs in
  `docs/agents/`, out of the always-loaded rules.
- **No committed handoff file.** Session-to-session orientation is a temporary
  artifact of the `handoff` skill, never tracked here. A tracked `HANDOFF.md`
  failed exactly once and predictably: it accreted durable operational content
  that went stale while cited from three other docs. If you are about to write
  next-session context into a tracked file, this is the rule that names the
  mistake.
- **No section numbers in heading titles** (`## Important Context`, not
  `## 1. Important Context`) — avoids renumbering churn and broken anchors.
  Explicit IDs stay: `ADR-0007`, audit finding `F10`, manuscript `§4.1`.
- **`docs/plans/` is active-only.** A plan exists only while it has open
  questions; on resolution it becomes an ADR and the plan file is deleted.

## Language

### Precision Scaling (Section 3.3)

**Soft quality signal**:
The blended [0,1] score $s_k^{(t)}$ combining a client's normalized gradient norm and normalized dataset size. Code: computed inline as `term` per formulation branch in `fedmaq.py`, not materialized as a standalone variable.
_Avoid_: intermediate signal, blended score

**Soft quality target**:
The bit-width value $\hat q_k^{(t)}$ derived from the soft quality signal, before Tier-1 clamping. Code: `q_hat` in `fedmaq.py`.
_Avoid_: soft quality function (former manuscript Ch3 wording; swept 2026-07-25, no occurrences remain)

**Formulation**:
One of five candidates (0-4) defining how the soft quality signal and soft quality target are computed: 0 = Resource-Only Hard Cap, 1 = Normalized Linear Weighted Sum, **2 = Normalized Multiplicative Scaling — frozen and shipped** (`conf/algorithm/fedmaq.yaml: formulation: 2`, [ADR-0012](docs/adr/0012-formulation-selection-and-the-iso-byte-amendment.md)), 3 = Gradient-Primary Data-Modulated, 4 = Threshold-Based Staged Rule. Code: `formulation` int param in `fedmaq.py`.
_Avoid_: "Alternative N" as a synonym for "Formulation N" (former manuscript Ch3 wording; swept 2026-07-25). _Avoid_: soft quality-target formulation (former Ch4 wording, redundant with "soft quality target"; swept 2026-07-25). _Avoid_: **"additive"** for Formulation 1 — the config spells it *linear* (`formulation1-linear-sum`); both spellings were live, and Ch2 §2.5 carried them in one paragraph (swept 2026-08-08, Ch2 fixed; **Ch6 consumed by pass 15's audit**, leaving Ch1/Ch3/Ch4 for the cross-chapter sweep). **Not every hit is a name.** `chapter_4.tex:406` uses "the additive Formulation 1" and "the additive form" contrastively against the multiplicative one inside a mathematical argument about what zeroing a weight does; that is a description of the operator, not a use of the label, and the sweep must re-derive each hit rather than replace on match. _Avoid_: **"non-adaptive control"** for Formulation 0 — it *is* adaptive at Tier 1, being memory-clamped like every other arm, and only Tier-2-blind; say **resource-only control** (repointed manuscript-wide 2026-08-08).

**Formulation constants**:
The tunable constants inside Formulations 1-4. **The manuscript's Greek symbols are canonical**; the config keys currently differ and are pending a rename:

| Canonical | Config key (pending rename) | Role |
|---|---|---|
| $\omega_1$, $\omega_2$ | `gamma1`, `gamma2` | Formulation 1 & 2 signal weights |
| $\kappa$ | `lambda_val` | Formulation 3 data modulator |
| $\tau_g$, $\tau_n$ | `tau_g`, `tau_n` | Formulation 4 thresholds (already aligned) |

Values agree on both sides ($\omega_1 = \omega_2 = 0.5$, $\kappa = 1.0$, $\tau_g = \tau_n = 0.5$); only the names differ, so nothing is numerically wrong today.

Decided 2026-07-25 by collision-checking every candidate symbol against all 40 `fedmaq-literature/kg/papers/*.md` nodes. $\gamma$ was ruled out: it appears 29 times in the corpus, including as FedProx's inexactness parameter $\gamma_k^t$, and FedProx is one of this thesis's own baselines. $\lambda$ was ruled out at 18 occurrences (AdaDQ-KD's KD loss weight, AdaGQ's step sizes), which is also why the config key carries the awkward `_val` suffix. $\kappa$ has zero corpus occurrences; $\omega$ has two, neither in a compared method. **Do not "fix" the manuscript toward the config keys** — that reintroduces the FedProx collision.

Pending work: rename `gamma1`/`gamma2` → `omega1`/`omega2` and `lambda_val` → `kappa` across `conf/algorithm/fedmaq*.yaml` (6 files), `src/fedmaq/core/quantization_planner.py`, and `tests/test_environment.py`, gated on a `scripts/golden_diff.py` run. Leave `docs/**/archive/` audits as historical records. Deferred from the 2026-07-25 Ch3 pass to keep that session manuscript-scoped; note that archived `multirun/` configs carry the old keys and would silently fall back to `.get()` defaults if replayed after the rename.

**Bit-width**:
A discrete value from the permissible set $\mathcal{Q} = \{1,2,3,4,5,6,7,8,16,32\}$ — never an arbitrary continuous integer.

**Tier 1 / Tier 2**:
FedMAQ's two-tier precision scaling design. Tier 1 is the hard feasibility constraint from client memory ($Q_k^{max}$), computed as a separate `min()` clamp in code, never blended into the soft quality signal. Tier 2 is the soft quality optimization (signal, target, formulation) layered on top and floored by Tier 1's cap.
_Avoid_: "three coequal dimensions of awareness" (resource, data, state) — resource (Tier 1) is structurally a hard clamp, not a third soft signal alongside data/state (Tier 2's two signals). The Ch4 rewording this entry once asked for has landed: `chapter_4.tex:106` now organizes the execution loop around that asymmetry "rather than around three symmetric awareness dimensions", and the phrase appears nowhere in the manuscript.

### Ablation Study (Section 4)

**State-only ablation**:
Ablation Configuration **3** — state (gradient-norm) awareness only drives Tier-2 quantization; server-side distillation is retained (as in configs 2-4). Names WHAT the arm configures. Both entries here read "Configuration 4" until 2026-08-08 (pass 15), which is the arm that removes *state* awareness and is therefore data-only — the mirror image. `conf/matrix/ablation.yaml` (`config3-no-data`, `config4-no-state`) and `chapter_4.tex:394`/`:402` are unanimous on Configuration 3.
_Avoid_: state-only-plus-distillation ablation, DynFed-core reference arm

**DynFed-style reference point**:
The role Ablation Configuration **3** plays in analysis (Ch4 §4, sec:ablation) — reproduces DynFed's core mechanism (gradient-norm-adaptive quantization, memory-capped, server-side multi-teacher distillation), absent DynFed's non-reproducible active teacher-selection. Explicitly framed as a comparison anchor, not a claimed win over DynFed itself (no public DynFed codebase exists to benchmark directly). Names WHY the arm exists.
_Avoid_: DynFed-core reference arm, **DynFed-style reference *arm*** (both collapse the what/why split these two entries exist to keep; `chapter_6.tex:40` carried the second and was fixed 2026-08-08. The manuscript's wording is *point* — `chapter_4.tex:402`, `chapter_1.tex:156`.)

### Communication Comparison (Section 4, `sec:metrics_communication`)

Three terms, one amendment. [ADR-0012](docs/adr/0012-formulation-selection-and-the-iso-byte-amendment.md)
replaced the primary criterion; these name what replaced it and what was demoted.
**A sentence that measures communication efficiency and leaves no slot for the first
two is defective even when every word in it is accurate** — that is the shape found
in Ch1 §1.3, Ch6 §6.1, Ch2 §2.2 and Ch3 §3.3.2, four passes running. **No sweep can
find it.** It is a shape, not a keyword: Ch3 §3.3.2 called the product "the true
communication cost" while the criterion sweep that covered Ch3 returned it as a
non-hit. Only a full read of the surface catches this class — which is why a sweep's
clean bill never licenses skipping it.

**Accuracy-vs-cumulative-MB curve**:
The **primary** communication-efficiency comparison. No free parameters. Mandated for
every run by the evaluation-metrics rule, and the axis on which every selection verdict
is read.
_Avoid_: single-round compression ratio as a stand-in (measures a different thing)

**Minimum common cumulative-MB budget**:
Where a scalar head-to-head is required: top-1 accuracy at $B = \min$ over the arms
compared of each arm's final cumulative MB. The budget is read off the data, chosen by
nobody — which is what keeps it free of the tunable parameter the $k$-consecutive rule
was rejected for.
_Avoid_: iso-byte budget, matched-byte budget, equal-expenditure budget (all appear in
pass notes as informal shorthand; none is the canonical term). Note that the $R = 100$
round budget equalizes **training** expenditure, not bytes — do not conflate the two.

**Bytes-to-target**:
Cumulative megabytes transmitted per client to reach a per-configuration target
accuracy. **Demoted 2026-08-06 from primary criterion to descriptor**, reported beside
the two above, never as the verdict. The target accuracy floor it rests on
(0.9 x FedAvg-at-equal-rounds) is **superseded**; cite it as such.
_Avoid_: **bits-to-target-accuracy**, **bits-to-accuracy**, **cumulative-MB-to-target**
— all three are live in the manuscript as non-canonical spellings of this one quantity.
A 2026-08-07 keyword sweep on `bytes-to-target` alone returned Ch2 clean; the chapter
was carrying the demoted quantity as primary under the `bits-` spelling. **Sweep the
whole variant set or the sweep proves nothing.**

**Rounds-to-target**:
A **different** quantity — the rounds term of the bytes-to-target product, not a
communication measure on its own. Pass 7 caught §5.2.4 reading it as a judgment on
total bytes. Pair it with per-round payload or do not cite it.
_Avoid_: **rounds-to-converge** (former Ch3 §3.3.2 wording, fixed 2026-08-07;
`chapter_4.tex:280` is canonical)

## Open items

**Last updated**: 2026-08-07.

The Ch1-Ch6 prose fixes logged here through 2026-07-25 were applied directly to
`fedmaq-manuscript` (main) and none remain. That is not a standing claim that the
manuscript is in sync: the sync passes of 2026-08-01 through 2026-08-07 each found
further drift, and the pinned **Manuscript sync log** Issue is the live record. Only the
`gamma`/`lambda_val` → `omega`/`kappa` rename above is tracked here, still deferred.
