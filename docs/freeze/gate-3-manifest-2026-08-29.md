# Gate 3 — Independent review: sanitized input manifest

Envelope: `fedmaq-experiments:docs/freeze/assurance-envelope-2026-08-29.json`
(#75). Gate: 3, **BLOCKED** pending a verifiable Terra High review.
Producer: assurance orchestrator. Prepared: 2026-08-29T22:59:04+08:00.

## Review package

- Candidate: `fedmaq-experiments@d804b7f2223fa92a8d2bcde803bec1501454faf5`.
  The local checkout used for preparation is `5d312d585313b8db56ef98ff2e93903d0e97eb97`,
  whose only post-candidate change is envelope/gate-evidence material.
- Bound revisions: `fedmaq-literature@1be87e08d232b8d47ec9c71e80922d7c8235bb5a`;
  `fedmaq-manuscript@5cc82393d639321c240897dbd7eeb728dfc82a02`.
- Review setting requested: **GPT-5.6 Terra High**, reasoning **high**. The
  available subagent execution did not expose an attested model receipt, so
  the setting is recorded as requested rather than falsely asserted as
  verified.
- Lens: independent standards/spec review of the candidate assurance package;
  check source-to-implementation fidelity, configuration and protocol claims,
  telemetry/byte-accounting semantics, pass-3 lifecycle invariants, evidence
  provenance, fail-closed gate behavior, and whether the stated scope boundary
  and invalidation ownership are sufficient. Do not infer experimental results
  from static artifacts, and do not convert an unresolved scientific question
  into a PASS.
- Required output: findings classified as critical, significant, minor, or
  verified invariant; each finding must cite a manifest entry and relevant
  line/path. The review output and disposition are recorded in
  `docs/freeze/gate-3-review-2026-08-29.md`.

## Stated exclusions

The following are not supplied as review inputs or are explicitly outside the
review lens:

- secrets and local configuration (`.env*`, credentials, tokens, personal
  settings), generated caches, bytecode, build products, PDFs, datasets,
  experiment outputs, and external GPU evidence;
- `fedmaq-analyses`, `fedmaq-presentations`, and
  `fedmaq-journal-article`, which are outside pipeline-readiness evidence;
- `fedmaq-literature/fedmaq-wiki/papers/*.md` except for the five method nodes
  listed below, and `fedmaq-experiments/docs/agents/**`,
  `docs/audits/**`, `docs/experiments/**`, and other unlisted freeze records;
- `fedmaq-manuscript/chapter_3.tex` section `sec:theo_bounds_quant`'s
  epsilon bound, Tier-2 rationale, and sum-of-squares argument. This is the
  declared #42 residual and is not adjudicated by this gate; the reviewer may
  use the rest of chapter 3 for claim-surface context;
- live issue bodies and the live envelope JSON are framing references, not
  hashed content entries. They can change after preparation and are therefore
  not silently treated as immutable review evidence. The final envelope's own
  content hash is recomputed after this artifact is added.

## Hash convention

`manifest_hash` is SHA-256 over the UTF-8 bytes of the exact text between
`BEGIN HASHED MANIFEST` and `END HASHED MANIFEST`, including line endings as
stored in this file and excluding all surrounding prose and this hash field.
Rows are sorted by repository and path. Each file hash is SHA-256 over the
exact Git blob at the revision shown in the row; the two prior gate evidence
rows intentionally identify their producing commits. This package contains identifiers and
hashes, not copied source contents; the thesis author should provide the
listed files from the pinned checkouts to Terra High.

Manifest hash: `c3593103abdbf1bf72057b2c3315793ef89497a52e3f22365a0620fbfeac7dc3`

<!-- BEGIN HASHED MANIFEST -->
| repository | revision | path | sha256 |
| --- | --- | --- | --- |
| fedmaq-experiments | d804b7f | CONTEXT.md | 7a030ffd3a4016b60789075fc7c2ecfc1ef86f1475e30758f4a0e8d079209207 |
| fedmaq-experiments | d804b7f | conf/algorithm/fedmaq.yaml | df71dd621aadd688c9eb79c8501fc9d7450a3f1e63a5996ec8ea9904bdb99e9f |
| fedmaq-experiments | d804b7f | conf/experiment/default.yaml | 4adb01c0295c3dcad1c8f3e8f0d8bc327a030097de09efc1e9b5f505482d98e2 |
| fedmaq-experiments | d804b7f | docs/adr/0019-quantizer-unbiasedness-and-the-l-infinity-exception.md | e7518b1647857c69efb06cdd1b244db718a5cb08d8e821b4c775ce2288ee4d26 |
| fedmaq-experiments | d804b7f | docs/adr/0020-secondary-byte-axis.md | 03c42bdcdf63649e95fa611dc336b45e10625750131f490cbd36ac7d28e8bf1e |
| fedmaq-experiments | d804b7f | docs/adr/0021-power-mean-formulation-family.md | c132c7374aaf65bf7ff244d154e6835ca7a1255ff2a56df26504eb2b3980f315 |
| fedmaq-experiments | d804b7f | docs/freeze/gate-1-evidence-2026-08-29.md | c66790ecd124ce4066d19a7a5c0c1b124b9f0a039ae6ec43ebf827d0c36b7b0a |
| fedmaq-experiments | 5d312d5 | docs/freeze/gate-2-evidence-2026-08-29.md | ff3c07562e74bf4c66a350a3ff06f0f7b00b25164ba12b78e33f8d353e5e00bc |
| fedmaq-experiments | d804b7f | docs/freeze/source_manifest.json | 97b5ddf99f7920e4d5c69ced4f09bc41b5e55102f91cdbd0ff79804b83c332b3 |
| fedmaq-experiments | d804b7f | justfile | a44fead19ae833a8eac24b4de5739adbfc03e57d69921624a186bb4975574ac2 |
| fedmaq-experiments | d804b7f | pyproject.toml | d96cdfc559bf60498cd67d05f6cb3885e6eafd9fd5a78c35cccc5f2cc0f145ef |
| fedmaq-experiments | d804b7f | scripts/check_freeze.py | 90ed8a6a2bf66b6015dd3f00679eb0529c5246c5d34daf3bbde580d12e35a0b2 |
| fedmaq-experiments | d804b7f | scripts/golden_diff.py | 99cfd4444317cdc0b14bd4aa05211a0b08785ea36383036922414e142de5aea3 |
| fedmaq-experiments | d804b7f | src/fedmaq/baselines/dadaquant_coder.py | e68e19d052c09d9465b8d5f6c7aa8684ae6ce0a95c271cc90df6f428045c068f |
| fedmaq-experiments | d804b7f | src/fedmaq/baselines/postprocess.py | a2bc68213727797e13bd3569b4ca7ba534b6c8b5d86dd4e4650f0e97d453396a |
| fedmaq-experiments | d804b7f | src/fedmaq/baselines/quantization.py | e257cc3caec1da1c9ce0f7d9e14d977071e30032424a7de57eb80bd89dfcbdff |
| fedmaq-experiments | d804b7f | src/fedmaq/baselines/transport.py | d83ba14be69fe280ec3bcdcb3b9db204a94bbf8c3cda68fbededc619e0bbfa9e |
| fedmaq-experiments | d804b7f | src/fedmaq/core/client_hooks/standard.py | 31fad31696922ff414cc44ad5d660d3b1726e1dc98ef0d8d978ef9691b39d17e |
| fedmaq-experiments | d804b7f | src/fedmaq/core/quantization_planner.py | a3546dd1a533157aefa83de83664007131eceadf0e27d25400180f5db48dd8a1 |
| fedmaq-experiments | d804b7f | src/fedmaq/core/strategy.py | 9e482366f955472d1a025de6e004d23c88c419691d4cfe6ef50d739b4954815f |
| fedmaq-experiments | d804b7f | src/fedmaq/core/strategy_hooks/feddistill.py | bc9296ab9eb9b54ac80c7b605559c5b3b80421246ec76939975eac9226e16907 |
| fedmaq-experiments | d804b7f | src/fedmaq/core/strategy_hooks/fedmaq.py | a0790b2df0c3091d645ceb435b7127cdc941ae9a12458c1fa080bbb7669c6382 |
| fedmaq-experiments | d804b7f | src/fedmaq/core/telemetry.py | d87742bd85f67c76d3b99ad7da4f2c6e685c5ec82ef5f980c4425e1dc8410da3 |
| fedmaq-experiments | d804b7f | src/fedmaq/simulation.py | 359d1c595751c5867cacbca44dbc08097948cca3744f1d8b460d851b43ac3dcd |
| fedmaq-experiments | d804b7f | tests/test_config_defaults.py | b562911960063ad32cb12fe0b71d1bcf3263dee0824cd9d01a57c14ad801f151 |
| fedmaq-experiments | d804b7f | tests/test_dadaquant_coder.py | 2606ffe10911d735670c836b7a8309f6d9e13276ee6d719dc072da4d02d38282 |
| fedmaq-experiments | d804b7f | tests/test_issue28_tier1_telemetry.py | 04ad510609a5a4101ff57da1eb2c71c65ddeb79eeb4d704d090c81cd6cd9ee11 |
| fedmaq-experiments | d804b7f | tests/test_loss_metrics_capability.py | 5569c4175a14c2cbbdb4746c6a805b56e2e3b1e136dc8f247705e003be5472c1 |
| fedmaq-experiments | d804b7f | tests/test_payload_capture.py | 5b63e831a16625219c985ded465934edaf2c6c1fc84d2ed7890207518b85b8c7 |
| fedmaq-experiments | d804b7f | tests/test_simulation_lifecycle.py | 505f014239eed2d25ba7ab13be5e3d6a1543c9e1058621356201c730dfadcba2 |
| fedmaq-experiments | d804b7f | tests/test_telemetry.py | 6df6b3d2b4989c7218560151b720a505d8f200827f4a123d0abbefb6fef6c96d |
| fedmaq-experiments | d804b7f | uv.lock | 7b24c4cedcce2298a5756546bbf67d401d807635997d340f3cf2d20722c940c7 |
| fedmaq-literature | 1be87e0 | fedmaq-wiki/methods/dadaquant.md | e368a68dcb53c61d61edc09dff20a97a9a0ab2adbef97e0a5ca4c0a1c9c3652a |
| fedmaq-literature | 1be87e0 | fedmaq-wiki/methods/feddistill.md | 6bf432ebc8b0eff86e741210607a555ada3950979f5f6cb30a263f877c6513a0 |
| fedmaq-literature | 1be87e0 | fedmaq-wiki/methods/fedkd.md | 8a2db663182a7fb306d7fba44aeb2d92dcfd6ec7c5683463390a726c479cb617 |
| fedmaq-literature | 1be87e0 | fedmaq-wiki/methods/fedmaq.md | c41366a08d80bc317b3351885aa5c4db8a0d914b2a8fcbb7e5f80e056a9d1e72 |
| fedmaq-literature | 1be87e0 | fedmaq-wiki/methods/fedpaq.md | 65d0c7f7005c0dc22423bc10cf1cae959b1cbc5d7ed87ca95e6da4f3cacc81c2 |
| fedmaq-manuscript | 5cc8239 | CONTEXT.md | 51afd2ef4e00de56f2173ad816aa9da7fb16283a7e757e764dc9baa41da07f66 |
| fedmaq-manuscript | 5cc8239 | chapter_1.tex | 8ed6bc45b2f0bdd5ea7504e141e007cfc4eca8ec1ba5ebc692d98860a3db4363 |
| fedmaq-manuscript | 5cc8239 | chapter_2.tex | 32775fec1d59b32291469613ea6029f0ab20003bd31bd71e39a627cb7ebacf20 |
| fedmaq-manuscript | 5cc8239 | chapter_3.tex | b27ad1c574a9e0480c2a0902b4ec42df95a4ae55aef7559e3b25ec181f071484 |
| fedmaq-manuscript | 5cc8239 | chapter_4.tex | 87d1da157105f92d32d6b9836c93d7126ad9b27ccb80c2fe26968a260356df6e |
| fedmaq-manuscript | 5cc8239 | chapter_5.tex | 6cc032d29ec255aa9797a30089220dc6419d38cea67f2f5066f779fd4dd1655f |
| fedmaq-manuscript | 5cc8239 | chapter_6.tex | 09f5a58b6d842f78138a67dd944ac147579847a235d8e9986134dd7935871ee7 |
| fedmaq-manuscript | 5cc8239 | myreferences.bib | 9ce333431e2333973d8eee79625bd79e6af2ec5ed01ebaf2a8db00c0e76e3a4c |
<!-- END HASHED MANIFEST -->

## Gate disposition

The five orchestrator-owned gate-3 evidence items are prepared here:
sanitized input manifest, manifest hash, stated exclusions, scope/lens, and
model/reasoning setting. The available internal review and its disposition are
recorded in `docs/freeze/gate-3-review-2026-08-29.md`; gate 3 remains BLOCKED
because the model setting is not independently attested.
