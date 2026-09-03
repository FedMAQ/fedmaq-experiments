# Pre-Dispatch Smoke Gate and Candidate Re-Declaration Protocol (#100)

This document records the pre-dispatch smoke gate protocol, assurance envelope invalidation analysis, and author re-declaration procedure required for closing Issue #100 prior to full-matrix dispatch (#93).

---

## 1. Assurance Envelope Invalidation Analysis

### Does regenerating the source manifest invalidate the current assurance envelope?
**Yes.**

- **Rationale**: The assurance envelope `docs/freeze/assurance-envelope-2026-08-29.json` was pinned to candidate commit `d804b7f`, covering 146 files under the pre-remediation architecture. The remediation tickets (#96, #97, #98, #99, #100) modified core runtime, telemetry, manifests, partitions, algorithms, and configurations within `scope.include`.
- **Impact**: Under ADR-0015 and the Gate 7 invalidation matrix (`docs/freeze/verify_gate_7.py`), any modification within `scope.include` invalidates Gates 1, 3, 4, and 6.
- **Protocol**: Re-declaring the candidate is an author action. The author executes the paste-ready script in Section 4 below to generate `docs/freeze/assurance-envelope-2026-09-04.json` pinned to the candidate commit.

---

## 2. Pre-Dispatch Smoke Commands (Author Executed)

Per repository rules, agents do not run experiments at any scale. The author executes the following 5 fast, CPU-only ($R=2$, $K=2$, `experiment.client_gpus=0`) cells in PowerShell:

**Re-run required before this gate can be called clear.** The evidence under `outputs/smoke/` was captured at commit `9bfca9e` and has never been re-captured at any later candidate. Section 1's own rule settles it without a file census: a single commit touching any path matched by `scope.include` invalidates this evidence, several such commits have landed since `9bfca9e`, and that holds regardless of the Cell 5 syntax fix. The authoritative scope is the `scope.include` patterns in `docs/freeze/source_manifest.json`; neither a file list nor a count is restated here, because a stale copy of either reads as authority, and a bare `git diff --name-only` does not filter by that scope. All 5 cells need re-capture at whatever candidate is current when the gate is run, not just Cell 5.

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
```

---

## 3. Automated Smoke Gate Evidence Verification

Once the smoke runs complete, execute the automated verifier:

```powershell
uv run python scripts/verify_smoke_evidence.py --smoke-dir outputs/smoke
```

This checks:
1. **Completion**: Core `validate_run_evidence()` passes with zero errors (#96 regression check).
2. **Split & Loader Provenance**: Manifest confirms `split="val"` and `loader_used="val"` for Cell 5, and `split="test"`, `loader_used="test"` for Cells 1–4 (#99).
3. **Partition Cache Provenance**: Run manifest binds `partition_cache.sha256` matching the canonical digest computed from the disk cache file (#99).
4. **Formulation Metadata**: Run manifest binds `formulation`, `p`, and `omega`.
5. **Tier-1 Binding Fraction**: `algorithm/fedmaq/tier1_binding_fraction` is logged and strictly non-zero (#97).
6. **Bit-Width Histograms**: Discrete histograms `q_count_{b}` and `q_hat_count_{b}` for $b \in \{2..8, 16\}$ are logged and sum to the sampled client count per round (#97).
7. **FedDistill+ Norm**: `algorithm/feddistill/global_logits_l2_norm` is logged and positive (#98).

---

## 4. Paste-Ready Candidate Re-Declaration Command (Author Action)

Execute the following paste-ready PowerShell command to generate `docs/freeze/assurance-envelope-2026-09-04.json` with the current git commit revision:

```powershell
uv run python -c "
import json, subprocess, sys
from pathlib import Path
from fedmaq.core.run_identity import config_sha256

commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
    raise SystemExit('working tree is dirty; this declaration would assert clean_tree falsely')
out = Path('docs/freeze/assurance-envelope-2026-09-04.json')
if out.exists():
    raise SystemExit(f'{out} already exists; refusing to overwrite a sealed envelope')
freeze = subprocess.run([sys.executable, 'scripts/check_freeze.py', '--check'], capture_output=True, text=True)
if freeze.returncode != 0:
    raise SystemExit('freeze certificate is not current; state_at_candidate would be false:\n' + freeze.stderr)
envelope = {
    'schema_version': 1,
    'envelope_id': 'fedmaq-pipeline-assurance-2026-09-04',
    'created_at': '2026-09-04',
    'created_by': 'thesis author (pre-dispatch candidate re-declaration)',
    'canonical_location': 'fedmaq-experiments:docs/freeze/assurance-envelope-2026-09-04.json',
    'purpose': 'Immutable, content-hashed record binding the pre-dispatch pipeline candidate across experiment, literature, and manuscript repositories. Covers the remediation closing #96, #97, #98, #99, and #100, and the subsequent pre-freeze assurance landing tracked at #92.',
    'specification': {
        'specification_issue': 'FedMAQ/fedmaq-experiments#92',
        'execution_issue': 'FedMAQ/fedmaq-experiments#100'
    },
    'content_hash': {'algorithm': 'sha256', 'value': None},
    'revision_vector': {
        'fedmaq-experiments': {
            'canonical_identity': 'https://github.com/FedMAQ/fedmaq-experiments',
            'revision': commit,
            'clean_tree': True,
            'role': 'remediated pipeline implementation, configuration, dispatch, tests'
        }
    },
    'candidate': {
        'repository': 'fedmaq-experiments',
        'revision': commit,
        'pinned_on': '2026-09-04',
        'pin_semantics': 'Pre-dispatch candidate: remediation #96-#100 plus the pre-freeze assurance landing (#92)'
    },
    'freeze_certificate': {
        'path': 'docs/freeze/source_manifest.json',
        'schema_version': 1,
        'state_at_candidate': 'current'
    }
}
digest = config_sha256(envelope)
envelope['content_hash']['value'] = digest
out.write_text(json.dumps(envelope, indent=2) + '\n', encoding='utf-8')
print(f'Wrote {out} pinned to {commit}')
print(f'content_hash {digest}')
"
```
