# FedMAQ Agent Changelog

Archive of session-to-session changelog entries for the FedMAQ thesis codebase.

**Policy (as of 2026-07-11):** going forward, only append entries for major
milestones (merged PRs, architecture shifts, phase completions) — not routine
per-session narrative, since claude-mem now covers that granularity. Existing
entries below are historical and not retroactively edited.

## Historical Entries

### 2026-07-15 — FedMAQ-Lite Sweeps Completed & Dual-Variant Architecture Setup (experiments)

Concluded all sweeps for the FedMAQ-Lite variant (SimpleCNN, ~2.16M params) and established the dual-variant architecture to support both lightweight and standard ResNet18GN models:

- **Soft-Voting Sweep (36 runs)**: Phase 1 (Ablation) confirmed soft-voting provides +2.48pp (α=0.1) and +3.96pp (α=1.0) accuracy gains at R50. Phase 2 (Grid Sweep) identified optimal parameters: `ew=4.0, pw=1.0` (α=0.1) and `ew=2.0, pw=0.5` (α=1.0) at R40.
- **Temperature Ablation (4 runs)**: Evaluated $T \in \{1.0, 2.0\}$. Proved that $T=1.0$ is critical under severe non-IID (α=0.1, −6.57pp accuracy penalty at $T=2.0$), while $T=2.0$ behaves acceptably only under homogeneous data (α=1.0). This provides empirical validation of the default $T=1.0$ choice for quantized edge distillation.
- **Dual-Variant Support (`fedmaq` / `fedmaq_lite`)**: Refactored model dispatching, strategy hooks, client hooks, and baselines to support both the standard ResNet18GN variant (`fedmaq`, ~11.17M params) and the lightweight SimpleCNN variant (`fedmaq_lite`, ~2.16M params). All prior sweep numbers are officially designated as the `fedmaq_lite` baseline in the documentation.

### 2026-07-14 — Formulation Refinements: Soft-Voting, EMA Student, and Gradient Smoothing (experiments)

Implemented three priority refinements to improve FedMAQ's distillation robustness under severe non-IID data heterogeneity, derived directly from the pilot study's per-round training analysis:

- **Quantization-Aware Soft-Voting**: Configured dynamic weights for ensemble distillation based on teacher prediction confidence (entropy) and quantization bit-width (precision). Added configuration keys `soft_voting`, `entropy_weight`, and `precision_weight`.
- **EMA Student Model**: Tracks student parameters with a configurable exponential moving average (`ema_student` and `ema_decay=0.99`), resolving the mid-to-late round accuracy regression observed in the pilot.
- **Gradient Norm Smoothing**: Integrates a per-client running gradient norm EMA (`grad_norm_ema` and `grad_norm_beta=0.7`) to stabilize precision assignments against mini-batch sampling noise.
- **Verification**: Created `tests/test_refinement_features.py` containing comprehensive unit tests verifying the correctness of all three features. Ran full simulation test suite with 87/87 passing tests.
- **Documentation**: Saved detailed performance analysis to `experiments/pilot-formulation-study-7-14/comments.md` and updated priority lists in `HANDOFF.md`.

### 2026-07-13 — 40-Round Baselines Sweep Completed; Proxy Dataset Shifted from 1600 to 3000 (experiments + manuscript)

Completed the 40-round CIFAR-10 evaluation sweep across the distillation-based algorithms (CFD, FedKD, FedMD, FedDistill, FedMAQ) using 50 clients and 0.2 client fraction.

- **FedMAQ** achieved **49.57%** test accuracy with a cumulative communication footprint of **1132.26 MB** (a **3.01x reduction** in communication versus FedDistill while gaining **9.49%** absolute accuracy).
- **FedDistill** reached **40.08%** accuracy at a huge cost of **3410.05 MB**.
- **FedKD** converged to **27.94%** accuracy at **33.79 MB** overhead.
- **CFD** used **1.22 MB** of bandwidth but got **21.25%** accuracy.
- **FedMD** reached **20.78%** accuracy at **47.73 MB** overhead.

Also shifted the server-side proxy dataset size ($D_{proxy}$) from 1600 to 3000 across all configurations, tests, and manuscript chapters (1, 4, 5) to reduce distillation noise and stabilize convergence under statistical heterogeneity.

### 2026-07-13 — CFD Chance-Level Bug Root-Caused; Partial Fix Applied, Not Yet Resolved (experiments)

Root-caused CFD's accuracy being pinned at chance level (~10%) across all soft-label quantization bit-widths (1/2/4/8/16, 40-round CIFAR-10 alpha=0.1 sweep). Instrumented `pre_aggregate_fit`/`_train_server_model` in `src/fedmaq/core/strategy_hooks/cfd.py` with per-round confidence/row-std/KL-loss logging: the averaged client soft-label targets were near image-independent (`targets_row_std`~0.01-0.02 across an 8-round test, vs. a healthy signal that should vary meaningfully per public image) despite individual client confidence being reasonably high and the server's KL loss dropping every round. This means the server was successfully learning to imitate a garbage, nearly-constant target — i.e., each client, trained from scratch each round on an extremely skewed (alpha=0.1) local partition, collapses to predicting its dominant local class regardless of input image, and averaging across clients just yields that round's sampled-client class-frequency prior instead of a real per-image signal.

Cross-checked against the source paper (`fedmaq-literature/markdown/sattler-2022-cfd/paper.md`, Sec. II-B step 2): the FD protocol CFD builds on requires participating clients to converge to the _same_ distilled theta each round (via seed-synchronized distillation to the downloaded soft-labels) before private training — functionally equivalent to a shared broadcast initialization, without transmitting weights. The port's client hook (`src/fedmaq/core/client_hooks/cfd.py`) was discarding this: `CFDFit.fit` never loaded the `parameters` argument (which Flower already populates with the persistent `server_model`'s weights via `CFDHook.aggregate_fit`) into `client.model`, instead training every client from an independent random init each round. Fixed by loading `parameters` via `set_model_parameters` at the top of `fit()`, matching the existing pattern in `feddistill.py`/`fedkd.py`/`standard.py`'s client hooks.

This fix is a genuine correctness improvement (paper-faithful weight sync was simply missing) but an 8-round CIFAR-10 alpha=0.1 re-test with the same instrumentation showed it was **not sufficient alone**: `targets_row_std` and test accuracy both stayed flat/noisy around chance (acc 9.28%-12.19%, row_std 0.0098-0.0193) with no upward trend through round 5. Leading hypothesis for the next session: under alpha=0.1, the few local CE epochs each round still overwrite the freshly-synced representation before it can generalize, re-collapsing predictions to the client's local class regardless of image content — candidates are reducing local CE dominance (fewer `local_epochs` or lower LR for CFD specifically) relative to the KL-distillation step, increasing `distill_epochs` so distillation converges more before CE resumes, or addressing round 1's zero-signal bootstrap (no distillation step at all in round 1, purely private CE on skewed classes, which may poison the very first server aggregate). `DEBUG_CFD` logger.info instrumentation left in place in `src/fedmaq/core/strategy_hooks/cfd.py` for the next session. Baseline registry status downgraded from `[Complete]` to `[In Progress]`.

### 2026-07-13 — FedKD Chance-Level Bug Fixed: Server-Side SVD Now Compresses Deltas (experiments)

Root-caused and fixed FedKD's accuracy being pinned at chance level (~10%/ln(10)) on CIFAR-10 regardless of round count. `FedKDHook._svd_compress_parameters` in `src/fedmaq/core/strategy_hooks/fedkd.py` was applying SVD rank-truncation directly to full aggregated weight matrices on both the download (`pre_configure_fit`) and eval (`pre_evaluate`) paths every round — full weight matrices aren't low-rank, so this collapsed `mean_rank_retained` to ~0.037 (near rank-1) almost immediately regardless of the energy ramp, independent of the upload-path compression (`FedKDCompressionHook.compress`) which already correctly compressed deltas.

Fix (commit `d8bcccd`): renamed to `_svd_compress_delta`, which tracks a client-side reference (`self._reference`) and SVD-compresses `parameters - reference` instead of raw weights, mirroring the upload path and matching the original paper's (Wu et al., 2022) gradient/update-compression design. `pre_evaluate` now reuses the reconstruction from `pre_configure_fit` instead of running a second, independent compression pass.

20-round CIFAR-10 verification (`experiment.total_rounds=20`, `client_fraction=0.2`, 10 clients/round): alpha=1.0 (homogeneous) converged cleanly to 24.19% accuracy / loss 2.017, down from chance-level 10%/ln(10)=2.303; alpha=0.1 (heterogeneous) reached 14.72%, noisier but no longer dead-flat. `mean_rank_retained` now correctly tracks the energy ramp (floors early when energy is low, climbs to 0.10-0.14 near tmax=0.9) rather than sitting fixed regardless of energy target.

Also surfaced and cleared a separate infra issue during this investigation: leaked Ray worker/actor processes from prior killed sweep launches accumulated across the session and saturated the 8GB GPU (down to 144 MiB free), causing cascading CUDA OOM errors independent of the FedKD bug. Killed via direct PID termination + `ray stop`; not yet root-caused why `pkill -f "scripts/run.py"` doesn't reach detached Ray actor subprocesses.

Remaining: 20 rounds/10 clients-per-round is still triage-scale, not benchmark-scale — a longer run (40-100 rounds) is needed before this counts as a confirmed FedKD result, especially to see if the alpha=0.1 (heterogeneous) case converges as cleanly as alpha=1.0 did.

### 2026-07-13 — Round 3 Final Audit Closes Out Grilling Thread (manuscript, literature, experiments)

Holistic cross-chapter read (Ch1-Ch6 + CONTEXT.md end-to-end) plus a self-check of Round 2's own edits, closing the multi-round grill-with-docs thread. Found and fixed instances of the "three coequal dimensions" pattern (banned since Round 1) that Round 2's chapter-by-chapter sweep missed in Ch1/Ch2/Ch3/Ch6; a self-contradiction in `chapter_4.tex:165` vs `:354`; an implicit-only bandwidth/compute-uniform claim in Ch1; a `kg/papers/` dangling reference from Round 2's FedDistill naming-collision fix that never touched the `papers/` tree; and a redundant `fd-faug.md`/`feddistill.md` content overlap. All fixes applied and pushed directly to `main` across three repos, per standing user instruction for this thread.

### 2026-07-12 — Ch1/2/5/6 Grilling Sweep Closes Out Manuscript-Code Drift (manuscript, literature, experiments)

Completes the grill-with-docs terminology/consistency pass across the whole manuscript (Ch3/Ch4 resolved in a prior session; this closes Ch1/2/5/6). All fixes applied and pushed directly to `main` across three repos.

- Split Objective 1 wording (`project-overview.md`, `fedmaq-experiments`): bandwidth/compute stay uniform, per-client memory is the heterogeneous axis feeding the Tier-1 hard clamp — Ch1/Scope had drifted from the fully-uniform phrasing.
- Corrected two mischaracterized baselines in Ch2 (`chapter_2.tex`): DAdaQuant adapts on a global-loss moving-average plateau (not raw elapsed time); LAQ-HC adapts on a data-quality/bandwidth flag function (not delay). Narrowed the research-gap argument to gradient-norm/optimization-geometry specifically.
- Hedged Ch5 (`chapter_5.tex`) preliminary-results prose from past to future/conditional tense — no runs were logged in `experiment_registry.md` yet.
- Fixed a naming collision in `fedmaq-literature`'s kg: `methods/feddistill.md` conflated Jeong et al. 2023 (the mechanism actually implemented as the "FedDistill" baseline) with Song et al. 2024 (a different de-biasing algorithm). Rescoped `feddistill.md` to Jeong's mechanism, split Song's into new `methods/feddistill-debias.md`, repointed cross-references.
- Logged all resolutions in `CONTEXT.md`'s Open Items section (`fedmaq-experiments`).
- No ADR-worthy architectural trade-offs surfaced; Ch6 and appendices had no drift.

### 2026-07-11 — CFD Baseline Port, Paper-Faithful (experiments)

Ports the last unported baseline (P11): CFD (Compressed Federated Distillation, Sattler
et al. 2022). Exchanges quantized soft-labels on the shared public proxy set instead of
weights/gradients; the manuscript (ch4 §Baselines) names it one of only two reproducible
Category-D (Hybrid Q+KD) comparators, so the 201-run grid was blocked on this port.

- **`core/softlabel_codec.py`** (new, pure numpy): `constrained_quantize` — b-bit uniform
  quantization onto the probability simplex via largest-remainder (Hamilton apportionment),
  exactly preserving `codes.sum(axis=1) == 2**b - 1` per row (paper eq. 10); reduces exactly
  to argmax one-hot at b=1 (eq. 11), verified rather than special-cased. `dequantize`,
  `encode_bytes` (delta + zlib, mirrors `postprocess.py`'s pattern), `codes_to_bytes`/
  `codes_from_bytes` for the `FitIns.config`/`parameters` transport channels.
- **Server `CFDHook`** (`core/strategy_hooks/cfd.py`): holds a persistent `server_model`
  (the hook is a long-lived object, unlike ephemeral simulated clients) refined every round
  via dual distillation on the aggregated client soft-labels; its predictions become next
  round's downstream target. `pre_aggregate_fit` returns the `(None, {})` tuple (not bare
  `None`) to bypass FedAvg weight-averaging the int64 soft-label codes. Round 1 skips the
  downstream broadcast (server model untrained, no delta reference) but clients still upload
  soft-labels, seeding dual distillation from round 1.
- **Client `CFDFit`** (`core/client_hooks/cfd.py`): fresh model init each round (paper's
  design — clients hold no persistent state); digests server labels via KL (round >= 2, gated
  like FedMD), trains private CE, returns quantized soft-label codes as `parameters` (not
  weights). Only the upstream delta-reference codes persist, via `client.state`
  (`Context.state`, now threaded into `GenericClient` itself, not just
  `get_compressor_hook`).
- **Circular-import fix:** `softlabel_codec.py` lives under `core/`, not `baselines/`
  (deviating from the original plan) — `client_hooks/cfd.py` needing it from `baselines/`
  would import that package's `__init__.py`, which needs `fedmaq.core.client.CompressionHook`,
  while `core.client` is still mid-import via its own `client_hooks` import. The codec has no
  `fedmaq` dependencies, so `core/` sidesteps the cycle cleanly.
- **Fidelity caveats (documented in-code):** zlib substitutes for CABAC/arithmetic coding;
  public proxy is the 1600-sample FedMAQ pool, not the paper's ~80k-sample STL-10.
- **Tests:** `tests/test_cfd.py` (new, 11 tests: quantizer math, delta/zlib codec, server
  dual-distillation across 2 rounds, FedAvg-bypass, client round-gating, upstream delta-state
  persistence); `tests/test_environment.py` (registry test now constructs `CFDHook` instead of
  asserting `NotImplementedError`); `tests/test_simulation.py` (2-round `run(cfg)` smoke, like
  FedDistill's); `tests/test_timing_golden.py` (2 new goldens for the client digest-phase
  timing contribution). 78 tests green, ruff clean, mypy no new errors. CPU end-to-end smoke
  on CIFAR-10 (2 rounds, 6 clients) completes and produces finite eval accuracy.
- **Docs:** `baseline_registry.md` row 14 → `[Complete]`; `HANDOFF.md` P11 → `[x]`.
