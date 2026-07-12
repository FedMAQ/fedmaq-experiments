# FedMAQ Thesis Domain

Communication-efficient federated learning via multi-adaptive quantization and knowledge distillation. Canonical glossary for terms shared across `fedmaq-experiments` (code) and `fedmaq-manuscript` (thesis) — resolves naming drift between the two.

## Language

### Precision Scaling (Section 3.3)

**Soft quality signal**:
The blended [0,1] score $s_k^{(t)}$ combining a client's normalized gradient norm and normalized dataset size. Code: computed inline as `term` per formulation branch in `fedmaq.py`, not materialized as a standalone variable.
_Avoid_: intermediate signal, blended score

**Soft quality target**:
The bit-width value $\hat q_k^{(t)}$ derived from the soft quality signal, before Tier-1 clamping. Code: `q_hat` in `fedmaq.py`.
_Avoid_: soft quality function (manuscript Ch3 wording — needs a rename pass to match code and this glossary)

**Formulation**:
One of five named alternatives (0-4) defining how the soft quality signal and soft quality target are computed: 0 = Resource-Only Hard Cap, 1 = Normalized Linear Weighted Sum, 2 = Normalized Multiplicative Scaling, 3 = Gradient-Primary Data-Modulated (current implementation default), 4 = Threshold-Based Staged Rule. Code: `formulation` int param in `fedmaq.py`.
_Avoid_: soft quality-target formulation (manuscript Ch4 wording, redundant with "soft quality target")

**Bit-width**:
A discrete value from the permissible set $\mathcal{Q} = \{1,2,3,4,5,6,7,8,16,32\}$ — never an arbitrary continuous integer.

**Tier 1 / Tier 2**:
FedMAQ's two-tier precision scaling design. Tier 1 is the hard feasibility constraint from client memory ($Q_k^{max}$), computed as a separate `min()` clamp in code, never blended into the soft quality signal. Tier 2 is the soft quality optimization (signal, target, formulation) layered on top and floored by Tier 1's cap.
_Avoid_: "three coequal dimensions of awareness" (resource, data, state) — Ch4 prose oversimplifies; resource (Tier 1) is structurally a hard clamp, not a third soft signal alongside data/state (Tier 2's two signals). Ch4 needs a rewording pass to state resource as Tier 1 and data/state as the two Tier 2 signals.

### Ablation Study (Section 4)

**State-only ablation**:
Ablation Configuration 4 — state (gradient-norm) awareness only drives Tier-2 quantization; server-side distillation is retained (as in configs 2-4). Names WHAT the arm configures.
_Avoid_: state-only-plus-distillation ablation, DynFed-core reference arm

**DynFed-style reference point**:
The role Ablation Configuration 4 plays in analysis (Ch4 §4, sec:ablation) — reproduces DynFed's core mechanism (gradient-norm-adaptive quantization, memory-capped, server-side multi-teacher distillation), absent DynFed's non-reproducible active teacher-selection. Explicitly framed as a comparison anchor, not a claimed win over DynFed itself (no public DynFed codebase exists to benchmark directly). Names WHY the arm exists.
_Avoid_: DynFed-core reference arm (not manuscript wording, drop entirely)

## Open items

Manuscript prose fixes needed (Ch3/Ch4), no code changes:

- Ch3 §3.3: rename "soft quality function" ($\hat q_k^{(t)}$) to "soft quality target" to match code and this glossary.
- Ch4 §4 (ablation study): reword "three dimensions of awareness" to state resource as Tier 1 (hard clamp) and data/state as the two Tier 2 signals, matching Ch3's actual mechanism.
- Ch3 §3.3: add a `[Alternative 0: Resource-Only Hard Cap]` bracketed heading to match the structure of Alternatives 1-4 (currently inline-only, despite Ch4's table treating it as a full 5th formulation row).
- Ch4 §4 (Ablation Study), line 356: delete the sentence proposing an "optional sixth arm substituting DynFed's recursive, inertial bit-width tracker" — decided not to pursue (no value added to the ablation); no recursive/inertial tracker code exists anywhere in `src/fedmaq/`, confirming the decision was never implemented. Also resolves the numbering collision (sentence called it a "sixth arm" while Configuration 6 already names something else).
- ~~Ch4 "Objective-4 bits-to-accuracy product" undefined~~ — false alarm, Objective 4 is defined in Ch1 §1.2 (benchmarking objective, matches `.claude/rules/project-overview.md`). No fix needed; just a cross-chapter reference invisible when reading Ch3/Ch4 alone.
