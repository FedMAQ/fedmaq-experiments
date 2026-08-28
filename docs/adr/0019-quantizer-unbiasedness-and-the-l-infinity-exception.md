# ADR-0019 — Unbiased stochastic rounding for FedPAQ/FedMAQ, and why the error-feedback path keeps l∞

**Status**: Accepted · 2026-08-27
**Related**: ADR-0006 (the golden-diff gate this required a re-capture against, DAdaQuant included); ADR-0018 (the `measure_bytes` seam this routes through unchanged)

## 2026-08-28 correction — DAdaQuant also requires l2 normalization

The original audit correctly repaired FedPAQ and FedMAQ but incorrectly treated
DAdaQuant's normalization as out of scope and called its quantizer correct. Hönig
et al. define Federated QSGD over the update divided by its Euclidean norm. The
local hook used `max(abs(d))`, so its stochastic rounding was unbiased only for a
different operator.

`DAdaQuantCompressionHook._scale` now returns `np.linalg.norm(d)` for every
adaptive level. A focused `[3,4]` regression distinguishes the required scale 5
from the l-infinity scale 4. This changes DAdaQuant's reconstructed updates, code
sparsity, primary bytes, and secondary 0-RLE/Elias-omega bytes; the replacement
campaign must rerun its affected cells, and the prior GPU golden cannot certify
the corrected hook. FedMAQ's error-feedback path remains the documented
l-infinity exception below.

## Context

`FedPAQCompressionHook`'s and `FedMAQPostProcessCompressionHook`'s `q>1` path rounded
deterministically (`np.round`) and normalized by l∞ (`max|d|`). Reisizadeh et al.
2020's Assumption 1 (which `FedPAQCompressionHook` implements) and `chapter_3.tex:84`'s
`Q_s(v_j) = ‖v‖₂ · sgn(v_j) · ξ_j(v,s)` both require `E[Q(x)] = x` under l2
normalization — deterministic rounding is biased, and l∞ is not the normalization
either source specifies. DAdaQuant's own quantizer (`_quantize_elem`) already rounded
stochastically, but as an independent inline implementation, not shared
code — the same one-concept-two-implementations seam ADR-0018 closed for byte
accounting (Issue #24, scoped originally to FedMAQ alone, corrected to cover FedPAQ on
the same grounds since both run on `FedPAQCompressionHook`).

Two other latent bugs shared the seam: `client_fn` is rebuilt fresh by Flower every
round (`ClientApp.__call__` invokes it per `Message`), so a compression RNG seeded only
at construction time (`simulation.py`) replayed an identical draw sequence every round
for a given client — stochastic rounding was unbiased *within* a round but not
independent *across* rounds, undermining the averaging argument the unbiasedness claim
depends on. And `DAdaQuantCompressionHook.__init__` fell back to an unseeded
`np.random.default_rng()` when no generator was passed, silently defeating
reproducibility rather than failing.

## Decision

**One shared `_stochastic_round(scaled, rng)`** (floor-plus-Bernoulli dithering, lifted
from DAdaQuant's original inline code) in `quantization.py`, called by FedPAQ, plain
(memoryless) FedMAQ, `FedMAQPostProcessCompressionHook`'s `q>1` path, and DAdaQuant.
**No unseeded fallback anywhere**: `_require_rng()` raises `ValueError` if a hook
reaches its stochastic branch with `rng=None`, rather than defaulting.

**Scale is l2 for FedPAQ/FedMAQ's `q>1` path, unconditionally l∞ for `q≤1`.** The
`q≤1` branch is pure sign quantization (`sgn(d)·scale`) with no `ξ_j` damping term —
feeding it l2 scale would inflate every coordinate by `~√d` (since `‖d‖₂ ≈ √d·avg|d|`),
not a subtle bias shift. This is a property of the sign-quantization *path*, not the
call site, so it is unconditional regardless of which scale the `q>1` path uses.
DAdaQuant now keeps its source-required l2 normalization explicit at its own call
site (`_scale`) rather than inheriting FedPAQ's q-dependent sign branch.

**`FedMAQPostProcessCompressionHook` (error feedback + diff-coding) is a deliberate
exception: it keeps l∞ scale, switching only its rounding to stochastic.** The issue
asked for l2 there too. An empirical check — run before committing to the l2 switch,
per the issue's own instruction not to treat it as settled without one — simulated the
error-feedback recurrence over 40 rounds on a 2,097,152-element tensor (matching
`SimpleCNN.fc1.weight`), isolating scale from rounding:

| scale | rounding | residual/delta @ round 39 |
| --- | --- | --- |
| l∞ | deterministic (today, pre-#24) | 2.4 |
| l∞ | stochastic (**shipped**) | 40, still climbing across sampled rounds |
| l2 | deterministic | 6.3 |
| l2 | stochastic (issue's literal ask) | **1.2×10⁶ — unbounded** |

Mechanism: under l2 scale the quantization step is `‖d‖₂/levels`, but per-coordinate
signal is only `~‖d‖₂/√d`. For `fc1.weight`, `√d≈1448` while `levels` maxes at 127
(q=8) and floors at 1 (q=2, FedMAQ's actual Tier-1 floor) — `levels ≪ √d` for every
config FedMAQ runs in practice, so the compressor fails the contraction property error
feedback needs (Karimireddy et al.), and each round's quantization noise becomes the
next round's residual, compounding rather than attenuating. l∞ keeps the step bounded
by the tensor's own peak magnitude regardless of dimension, so it stays stable — the
40×-and-climbing figure above is real elevated risk relative to today's 2.4×, not
nothing, but it is what shipped, confirmed with the repository owner, because the l2
alternative is a live production hazard, not a theoretical one. **Extending l2 to this
path needs its own ticket with a redesigned feedback scheme** (e.g. clipping, a decayed
residual, or a dimension-aware level floor) — the reasoning above is inline in
`postprocess.py` for the next session that considers it.

**The compression RNG is reseeded every round**, not just at `client_fn` construction:
`StandardFit.fit()` reseeds any `compressor_hook` exposing a `.rng` attribute from
`(seed, partition_id, server_round)` — a tuple seed, deliberately not the same
`seed + pid*100_000 + round` integer arithmetic `quantization_planner.py:268` uses for
dataloader seeding, so the two draw from provably independent streams rather than
merely different-looking ones. Both `seed` and `server_round` are read via required key
access, not `.get(..., default)`: a silent fallback for either would reseed the
compression stream out of step with the rest of the run under a fixed seed, which is
the exact bug this ADR closes, reintroduced through the back door.

## Consequences

- **Realized code sparsity is a real, load-bearing behavior change on the memoryless
  l2 path**, not the error-feedback path (which is unchanged in scale). Measured on the
  shipped `FedPAQCompressionHook` at `fc1.weight` size (d=2,097,152): nonzero-code
  fraction is ~0.05% at q=2, ~0.38% at q=4, ~7.0% at q=8. `client/avg_q` telemetry is
  **unchanged** by this — it is the server-assigned bit-width (`fit_metrics["q"]`),
  never a realized-code statistic; nothing currently instruments the realized
  distribution as its own metric.
- **Golden-diff re-baseline scope is `fedpaq`, `fedmaq`, and `dadaquant`** — wider than
  the issue named. DAdaQuant reorders RNG-consuming operations exactly like FedPAQ/
  FedMAQ do (previously seeded once at construction, now reseeded per round through the
  same `StandardFit.fit()` code path), so per ADR-0006 its baseline is stale too, even
  though only fedpaq/fedmaq were named in the original acceptance criteria.
- Any future baseline reaching a stochastic-rounding branch must route through
  `_stochastic_round`/`_require_rng`, never inline its own floor-plus-Bernoulli or an
  unseeded fallback — that duplication is exactly the seam this ADR closed.
- `chapter_3.tex`'s `ε = min(d/s², √d/s)` bound, Tier-2 rationale (`:131`), and
  `Σₖpₖ²` argument (`:238`) are **not yet formally re-checked against the shipped
  operator** — the realized nonzero-code-fraction data above (~0.05%–7.0% depending
  on q) was measured to inform that check, not to complete it. Whoever next touches
  `chapter_3.tex §sec:theo_bounds_quant` should treat that measurement as the input,
  not the answer, and record a real pass/fail rather than inferring one from this
  ADR. No manuscript edit is expected to be *needed*; confirming that is still open.
