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

All previously logged Ch1-Ch6 prose fixes and nits have been applied directly to `fedmaq-manuscript` (main). None remain outstanding as of this session.
