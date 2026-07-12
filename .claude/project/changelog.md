# FedMAQ Agent Changelog

Archive of session-to-session changelog entries for the FedMAQ thesis codebase.

**Policy (as of 2026-07-11):** going forward, only append entries for major
milestones (merged PRs, architecture shifts, phase completions) — not routine
per-session narrative, since claude-mem now covers that granularity. Existing
entries below are historical and not retroactively edited.

## Historical Entries

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

### 2026-07-09 — Literature OKF Restructure (literature)

Restructured `fedmaq-literature` from a vector-RAG pipeline into an Open Knowledge Format (OKF) knowledge graph: raw markdown layer (citable) plus curated OKF nodes, no vector store. Converted all canon papers and migrated the paper registry; removed the RAG stack and its tests.

### 2026-07-09 — Deep Cleanup/Refactor + FedDistill+ Port (experiments)

Phased refactor: extracted a testable simulation entrypoint, fixed several correctness bugs (a gradient-norm architecture mismatch, an unreachable-NaN quantization case, missing bounds clamping), deduplicated shared strategy logic, replaced string-dispatch with per-algorithm hook classes, and ported the FedDistill+ baseline. Full test suite green.

### 2026-07-09 — Manuscript Alignment: Proxy Pool Size, Discrete Bit-Widths, FedDistill Spec (experiments + manuscript)

Reconciled the codebase against updated manuscript chapters (manuscript is canon): corrected the proxy pool size default and a remainder-allocation bug, snapped FedMAQ bit-widths to the manuscript's discrete set, fixed compressor semantics for wide bit-widths, and updated baseline registry notes. Manuscript edits made in parallel.

### 2026-07-09 — CLAUDE.md Relocation to Repo Root and AGENTS.md Removal

Moved CLAUDE.md to repo root in experiments and manuscript; slimmed always-imported rules to a small core set with the rest routed via a table; removed the now-redundant AGENTS.md from experiments.

### 2026-07-09 — Cursor to Claude Code Migration (experiments + manuscript)

Migrated experiments and manuscript from Cursor config to Claude Code equivalents (rules, skills, project registries, slash commands); deleted the old Cursor config. Other repos not yet migrated at this point.

### 2026-07-03 — Codebase Hardening, Optimization & Correctness (Refactor Session)

Fixed partition-resolution performance, ensemble evaluation memory use, stochastic-rounding seeding, telemetry import robustness, and device-config edge cases. Full test suite passed.

### 2026-07-03 — Core Codebase Refactoring & Hardening

Extracted algorithm-specific strategy logic into modular hooks, fixed a quantization-level/bit-count estimation bug, added caching and vectorized metrics, and tightened typing. Full test suite passed.

### 2026-07-02 — Workspace Agent Context Pruning & Slash Workflows Setup

Pruned redundant content from the cross-repo handoff doc, added slash workflows for manuscript alignment/baseline addition/benchmark runs, and consolidated hyperparameter rules into the manuscript-alignment rule.

### 2026-07-02 — Manuscript Alignment Rule Creation and Test Expansion

Created the manuscript-alignment rule (hyperparameters, quantization formulations, simulated delays, test requirements) and expanded formulation test coverage.

### 2026-07-02 — Uniform System Simulation and Preliminary Config Updates

Moved to uniform bandwidth/compute across clients, decoupled simulated client and server timing, and added per-algorithm compute-penalty/pretrain-delay modeling. Resolved a test flakiness issue and validated with a multi-algorithm benchmark run.

### 2026-07-01 — Manuscript Table 4.1 Hyperparameter Synchronization and Gitignore Cleanup

Synchronized hyperparameters between the manuscript and codebase, implemented learning-rate decay, corrected the default proxy dataset size, and cleaned up gitignore/untracked logs.

### 2026-07-01 — Full Manuscript Audit (Ch. 1--4) and Codebase Hardening

Audited chapters 1-4 against the codebase and fixed several bugs (heterogeneous compute speed, unseeded stochastic rounding, config separation, dead config, stale defaults, unsafe device resolution). Flagged a manuscript hyperparameter table error and confirmed FedDistill/CFD as pending stubs at the time.
