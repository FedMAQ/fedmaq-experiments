# FedMAQ Thesis Domain

Multi-adaptive quantization and knowledge distillation for memory-constrained federated learning under non-IID data. Canonical glossary for terms shared across `fedmaq-experiments` (code) and `fedmaq-manuscript` (thesis) — resolves naming drift between the two.

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
_Avoid_: "Alternative N" as a synonym for "Formulation N" (former manuscript Ch3 wording; swept 2026-07-25). _Avoid_: soft quality-target formulation (former Ch4 wording, redundant with "soft quality target"; swept 2026-07-25)

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
Ablation Configuration 4 — state (gradient-norm) awareness only drives Tier-2 quantization; server-side distillation is retained (as in configs 2-4). Names WHAT the arm configures.
_Avoid_: state-only-plus-distillation ablation, DynFed-core reference arm

**DynFed-style reference point**:
The role Ablation Configuration 4 plays in analysis (Ch4 §4, sec:ablation) — reproduces DynFed's core mechanism (gradient-norm-adaptive quantization, memory-capped, server-side multi-teacher distillation), absent DynFed's non-reproducible active teacher-selection. Explicitly framed as a comparison anchor, not a claimed win over DynFed itself (no public DynFed codebase exists to benchmark directly). Names WHY the arm exists.
_Avoid_: DynFed-core reference arm (not manuscript wording, drop entirely)

## Open items

**Last updated**: 2026-08-07.

The Ch1-Ch6 prose fixes logged here through 2026-07-25 were applied directly to
`fedmaq-manuscript` (main) and none remain. That is not a standing claim that the
manuscript is in sync: the sync passes of 2026-08-01 through 2026-08-07 each found
further drift, and `docs/STATUS.md` § "Manuscript Sync" is the live log. Only the
`gamma`/`lambda_val` → `omega`/`kappa` rename above is tracked here, still deferred.
