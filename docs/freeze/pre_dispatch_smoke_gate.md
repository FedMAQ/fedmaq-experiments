# Pre-Dispatch Smoke Gate and Candidate Re-Declaration Protocol (#100)

This document records the reusable pre-dispatch smoke and assurance-envelope
re-declaration procedure established for Issue #100. It is not a live readiness
record.

Before proposing any stage command, read the relevant execution Issue and its dated
assurance envelope. If they name a sealed envelope for the exact candidate, preserve
and verify that evidence, then proceed through the matrix handoff. Use this document's
smoke and re-declaration steps only for a new, missing, or invalidated candidate.

---

## 1. Assurance Envelope Invalidation Analysis

### Does regenerating the source manifest invalidate the current assurance envelope?
**Yes.**

- **Rationale**: The assurance envelope `docs/freeze/assurance-envelope-2026-08-29.json` was pinned to candidate commit `d804b7f`, covering 146 files under the pre-remediation architecture. The remediation tickets (#96, #97, #98, #99, #100) modified core runtime, telemetry, manifests, partitions, algorithms, and configurations within `scope.include`.
- **Impact**: Under ADR-0015 and the Gate 7 invalidation matrix (`docs/freeze/verify_gate_7.py`), any modification within `scope.include` invalidates Gates 1, 3, 4, and 6.
- **Protocol**: Re-declaring the candidate generates a fresh dated assurance envelope pinned to the candidate commit with the Section 4 script, which refuses to overwrite an existing envelope.

---

## 2. Pre-Dispatch Smoke Commands (Author Executed)

Per repository rules, agents do not run experiments at any scale. The author executes the following 6 fast, CPU-only ($R=2$, $K=2$, `experiment.client_gpus=0`) cells in PowerShell:

Candidate-specific smoke status lives in the relevant execution Issue and dated
assurance envelope. The `9bfca9e` evidence is historical and must not be used to
qualify a new candidate. For a new or invalidated candidate, recapture all six cells
at the exact commit and record the result in a new envelope. Any source or config
change after capture invalidates the gate and requires a new capture.

```powershell
# 1. FedMAQ (Calibrated c_unit=1024, post-processing enabled, power-mean base)
uv run python scripts/run.py experiment=ci experiment.client_gpus=0 algorithm=fedmaq algorithm.post_process=true hydra.run.dir=outputs/smoke/fedmaq

# 2. Power-Mean Arm (p=0, omega=0.5 power-mean formulation)
uv run python scripts/run.py experiment=ci experiment.client_gpus=0 algorithm=power_mean hydra.run.dir=outputs/smoke/power_mean

# 3. FedDistill+ (Masked logits target & global logits norm)
uv run python scripts/run.py experiment=ci experiment.client_gpus=0 algorithm=feddistill hydra.run.dir=outputs/smoke/feddistill

# 4. Ablation Arm (Configuration 3: fedmaq_no_data)
uv run python scripts/run.py experiment=ci experiment.client_gpus=0 algorithm=fedmaq_no_data hydra.run.dir=outputs/smoke/ablation_no_data

# 5. Validation Split Check (Stage-A selection split on validation loader)
# NOTE: was `+split=val`. Commit dcf3596 added a base `split: test` key to conf/config.yaml,
# so `+split=val` now fails with a Hydra "already at 'split'" error (+ requires the key be absent).
uv run python scripts/run.py experiment=ci experiment.client_gpus=0 split=val protocol_stage=matched_tuning algorithm=fedmaq hydra.run.dir=outputs/smoke/val_split

# 6. V2 candidate (ADR-0016 2026-09-25: fedmaq_no_kd, post-processing enabled)
uv run python scripts/run.py experiment=ci experiment.client_gpus=0 algorithm=fedmaq_no_kd algorithm.post_process=true hydra.run.dir=outputs/smoke/no_kd_post
```

---

## 3. Automated Smoke Gate Evidence Verification

Once the smoke runs complete, execute the automated verifier:

```powershell
uv run python scripts/verify_smoke_evidence.py --smoke-dir outputs/smoke
```

This checks:
1. **Completion**: Core `validate_run_evidence()` passes with zero errors (#96 regression check).
2. **Split & Loader Provenance**: Manifest confirms `split="val"` and `loader_used="val"` for Cell 5, and `split="test"`, `loader_used="test"` for Cells 1–4 and 6 (#99).
3. **Partition Cache Provenance**: Run manifest binds `partition_cache.sha256` matching the canonical digest computed from the disk cache file (#99).
4. **Formulation Metadata**: Run manifest binds `formulation`, `p`, and `omega`.
5. **Tier-1 Binding Fraction**: `algorithm/fedmaq/tier1_binding_fraction` is logged and strictly non-zero (#97).
6. **Bit-Width Histograms**: Discrete histograms `q_count_{b}` and `q_hat_count_{b}` for $b \in \{2..8, 16\}$ are logged and sum to the sampled client count per round (#97).
7. **FedDistill+ Norm**: `algorithm/feddistill/global_logits_l2_norm` is logged and positive (#98).

---

## 4. Candidate Re-Declaration

Run `scripts/declare_assurance_envelope.py` from the local workspace, not JupyterHub. It needs `fedmaq-literature` and `fedmaq-manuscript` as sibling checkouts and network access to each repository's published `main`. It refuses a dirty tree, a HEAD that is not origin/main, or a stale freeze certificate. It writes a non-overwriting `docs/freeze/assurance-envelope-<YYYY-MM-DD>[-N].json`, sealed with its own `content_hash`, which `tests/test_freeze.py` recomputes.

Each envelope states its own candidate. Pass the purpose, issues, pin semantics, and declarer for this candidate; an earlier campaign's text never carries over:

```powershell
uv run python scripts/declare_assurance_envelope.py `
  --purpose "<what this envelope certifies>" `
  --specification-issue "FedMAQ/fedmaq-experiments#<registration issue>" `
  --execution-issue "FedMAQ/fedmaq-experiments#<dispatch issue>" `
  --pin-semantics "<why this commit is the candidate>" `
  --created-by "<declarer>"
```

Envelopes up to 2026-09-17 came from the inline recipe this section carried before; the script produces the same fields and hash.
