# Gate 4 — Local verification at the candidate: evidence

Envelope: `docs/freeze/assurance-envelope-2026-08-29.json` (#75). Gate: 4.
Producer: assurance orchestrator. Run date: 2026-08-29 (Asia/Manila).

## Revision binding

- Candidate pin: `d804b7f2223fa92a8d2bcde803bec1501454faf5`.
- Bound revision vector: `fedmaq-experiments@d804b7f2223fa92a8d2bcde803bec1501454faf5`,
  `fedmaq-literature@1be87e08d232b8d47ec9c71e80922d7c8235bb5a`,
  `fedmaq-manuscript@5cc82393d639321c240897dbd7eeb728dfc82a02`.
- Executed checkout: detached `d804b7f2223fa92a8d2bcde803bec1501454faf5`.
- `main` was at `5d312d585313b8db56ef98ff2e93903d0e97eb97` when this gate was
  prepared; `git merge-base --is-ancestor d804b7f 5d312d5` returned success.
- The post-candidate delta is limited to the envelope and gate-evidence files,
  so the exact candidate revision is the revision tested by this run.
- The candidate commit itself is clean. At execution time, the only working-tree
  addition was the untracked gate-3 manifest prepared for this package; it is
  under `docs/freeze/`, outside `source_manifest.json`'s `scope.include`, and
  cannot affect the freeze check, Ruff, or pytest inputs.
- Command: `just check`.
- Evidence digest: `3f1c20fa8c87fdaf03bcf628debe57c23425bac07283dd343f6cd8e7952f25b1`.

The evidence digest is SHA-256 over this file's UTF-8 bytes with the complete
- `Evidence digest: ...` line replaced by the literal text `- Evidence digest: <excluded>`. No
other bytes are normalized or omitted.

## Full command output

```text
uv run python scripts/check_freeze.py --check
docs\freeze\source_manifest.json is current
uv run ruff check .
All checks passed!
uv run python -m pytest
============================= test session starts =============================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Quirora\Documents\GitHub\BSMSCS Thesis\fedmaq-experiments
configfile: pyproject.toml
testpaths: tests
plugins: hydra-core-1.3.4
collected 415 items

tests\test_agent_context.py .                                            [  0%]
tests\test_aggregation_order.py ...                                      [  0%]
tests\test_analysis.py .....                                             [  1%]
..........................                                               [ 12%]
.........................                                                [ 19%]
tests\test_architecture_pass3.py ..                                      [ 20%]
tests\test_audit_study1_artifacts.py ..                                  [ 20%]
tests\test_cfd.py ............                                           [ 23%]
tests\test_client_manager.py .......                                     [ 25%]
tests\test_common.py ..............                                      [ 28%]
tests\test_config_and_dispatch.py ...................................... [ 37%]
........................                                                 [ 44%]
tests\test_config_defaults.py .........                                  [ 46%]
tests\test_dadaquant_coder.py .....................                      [ 51%]
tests\test_fedkd_compression.py .......                                  [ 53%]
tests\test_freeze.py ......                                              [ 54%]
tests\test_issue28_tier1_telemetry.py .........                          [ 56%]
tests\test_issue47_payload_bytes.py ....                                 [ 57%]
tests\test_issue49_upload_report.py ......                               [ 59%]
tests\test_issue50_download_report.py ..........                         [ 61%]
tests\test_loss_metrics_capability.py .....                              [ 62%]
tests\test_models.py ....                                                [ 63%]
tests\test_models_and_algorithms.py ........                             [ 72%]
............                                                             [ 75%]
tests\test_package_import.py .                                           [ 75%]
tests\test_payload_capture.py ..........................                 [ 81%]
tests\test_postprocess.py ...........                                    [ 84%]
tests\test_refinement_features.py ....                                   [ 85%]
tests\test_report_schema.py ......                                       [ 86%]
tests\test_run_identity.py .....................                         [ 92%]
tests\test_simulation_lifecycle.py .......                               [ 93%]
tests\test_strategy_hook_invariants.py ..                                [ 94%]
tests\test_telemetry.py .....                                            [ 95%]
tests\test_timing_golden.py ........                                     [ 97%]
tests\test_training_skeleton.py ..                                       [ 97%]
tests\test_transport.py .........                                        [100%]

============================== warnings summary ===============================
tests/test_config_and_dispatch.py::test_run_cfg_smoke_fedavg
  C:\Users\Quirora\Documents\GitHub\BSMSCS Thesis\fedmaq-experiments\.venv\Lib\site-packages\ray\_private\worker.py:2051: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default).
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================= 415 passed, 1 warning in 247.48s (0:04:07) ==================
```

## Disposition

Gate 4: **PASS**. The freeze certificate check, Ruff, and all 415 tests passed.
The single warning is the Ray `FutureWarning` shown above and did not fail the
check. This is local verification evidence only; it does not declare pipeline,
evidence, or results freeze.
