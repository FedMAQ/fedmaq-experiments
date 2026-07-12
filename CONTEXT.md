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

### Resolved this session — Round 3 final audit (holistic cross-chapter pass)

- **"Three dimensions" pattern (banned by this glossary) recurred outside Ch4** — Round 1 fixed it in Ch4 (`:112`, `:121`) but a chapter-by-chapter sweep in Round 2 missed identical instances in `chapter_1.tex:211`, `chapter_2.tex:429,442,425-438`, `chapter_3.tex:172`, `chapter_6.tex:40`. All reworded to Tier 1 hard clamp (memory) + Tier 2 two soft signals (data, state), matching Ch4's canonical phrasing. **Applied** (`fedmaq-manuscript` main).
- **`chapter_4.tex:165` self-contradiction**: used glossary-banned terms "DynFed-core reference arm" / "state-only-plus-distillation" while the same chapter's `:354` correctly said "DynFed-style reference point" / "state-awareness-only ablation (Configuration 4)". Reworded `:165` to match `:354`. **Applied.**
- **Objective 1's bandwidth/compute-uniform claim was implicit, never explicit in the manuscript**: `chapter_1.tex:164` only ever varied memory, never stated bandwidth/compute are held uniform by design. Added an explicit clause. **Applied.**
- **`fd-faug.md` duplicated `feddistill.md`'s FD mechanism/formula** instead of linking to it, undercutting the stated file split. Trimmed to a one-line summary + link. **Applied** (`fedmaq-literature` main).
- **Dangling FedDistill naming-collision reference in `kg/papers/`**: Round 2's naming-collision fix (see below) repointed `kg/concepts/` and `kg/methods/` but never touched `kg/papers/`. `jeong-2023-feddistill-aug.md:49` listed Song et al. 2024's FedDistill as a communication-reducing refinement of Jeong's mechanism — false, per Song's own paper ("No Communication Reduction"). Removed the false link, added a disambiguation note. `song-2024-feddistill.md` frontmatter `baseline: FedDistill` and `kg/papers/index.md`'s matching annotation reworded to clarify it's a name collision, not the implemented FedMAQ baseline. **Applied.**

- ~~Ch4 "Objective-4 bits-to-accuracy product" undefined~~ — false alarm, Objective 4 is defined in Ch1 §1.2 (benchmarking objective, matches `.claude/rules/project-overview.md`). No fix needed; just a cross-chapter reference invisible when reading Ch3/Ch4 alone.

### Resolved this session — Ch1/Ch2/Ch5/Ch6 sweep

- **Objective 1 uniform-vs-heterogeneous contradiction**: `project-overview.md` said fully "uniform system parameters"; Ch1/Scope assume heterogeneous per-client memory (Tier-1 hard clamp). Resolved as: bandwidth/compute uniform, memory heterogeneous. **Applied** (`.claude/rules/project-overview.md`, `fedmaq-experiments` main).
- **Ch2 DAdaQuant mischaracterization** (`chapter_2.tex:258-259`): claimed DAdaQuant adapts on raw elapsed time; source paper (Hönig et al. 2022) shows it reacts to a moving-average global-loss plateau. Narrowed the research-gap argument to gradient-norm/optimization-geometry specifically, which neither DAdaQuant nor LAQ-HC captures. **Applied.**
- **Ch2 LAQ-HC mislabeled "delay-adaptive"** (`chapter_2.tex:258`, `:280`): source paper (Cui et al. 2026) shows selection via a data-quality/bandwidth flag function, not delay/latency. Renamed to "quality/bandwidth-adaptive." **Applied.**
- **Ch5 asserted unexecuted results as completed** (`chapter_5.tex:24,45-51`): past-tense claims ("we executed," "we evaluated") had no backing in `experiment_registry.md` (zero logged runs). Reworded to future/conditional tense until real runs are logged. **Applied.**
- **FedDistill naming collision in `fedmaq-literature` kg**: `kg/methods/feddistill.md` conflated Jeong et al. 2023's per-label-logit mechanism (the actually-implemented baseline) with Song et al. 2024's group-distillation/de-biasing mechanism. Rescoped `feddistill.md` to Jeong's mechanism; split Song's into new `kg/methods/feddistill-debias.md`; repointed cross-references. **Applied** (`fedmaq-literature` main).
- Ch1 Objective 2 wording gap (objective statement omits data/state awareness, present only in \S1.3.2 prose): reviewed, left as-is — intentional abstraction.
- Ch6: placeholder-only, no drift found.
- Appendices A/B: boilerplate, not referenced by Ch1/2/5/6, no drift.
- No ADR-worthy architectural trade-offs surfaced in this sweep.

### Resolved this session (applied directly to `fedmaq-manuscript`, main branch)

- **"Soft quality function" -> "soft quality target"** (`chapter_3.tex:202`): renamed to match code and this glossary. **Applied.**
- **"Three dimensions of awareness" reword** (`chapter_4.tex:112`, `:121`): restated as Tier 1 hard clamp (resource) vs Tier 2's two soft signals (data, state), matching Ch3's actual mechanism. **Applied.**
- **`[Alternative 0: Resource-Only Hard Cap]` heading** (`chapter_3.tex`, after the Candidate Soft Quality Formulations intro paragraph): added for structural parity with Alternatives 1-4. **Applied.**
- **Sixth-arm sentence deleted** (`chapter_4.tex`, formerly line 357): removed the unpursued "DynFed recursive, inertial bit-width tracker" ablation-arm proposal; no such code exists in `src/fedmaq/`. **Applied.**

- **Rounding notation** (`chapter_3.tex:230`, $q_{mid} = \lfloor (q_{max}+q_{min})/2 \rceil$): mixed floor-left/ceiling-right bracket is standard round-to-nearest notation, not an error — no precedent found in `fedmaq-literature` kg (checked DynFed, QSGD, DAdaQuant), so added a defining footnote rather than replacing the notation. **Applied.**
- **`c_unit = 512` MB derivation** (`chapter_3.tex:182`): flagged in prior session as an unresolved gap; re-checked and the existing prose already ties it qualitatively to activation maps/gradient buffers/scratch space beyond the 1.4MB/bit-level payload figure — no formula needed. **No fix required, false alarm.**
- **Dual `c_k` values in hyperparameter table** (`chapter_4.tex:195`): one row conflated `$\mathcal{U}(2048,16384)$` (main grid) and `8192` (uniform-memory control arm). Split into two labeled rows so the table itself (not just surrounding prose) distinguishes the two regimes. **Applied.**
