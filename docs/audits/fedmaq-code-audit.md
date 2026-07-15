# FedMAQ Code-Level Audit — Mechanism Paths + MobileNetV2GN

**Last updated:** 2026-07-16
**Auditor:** Claude (Opus 4.8), grill-with-docs session
**Lens:** code craftsmanship + FL engineering conventions. **Distinct** from
[`fedmaq-audit.md`](fedmaq-audit.md) / [`fedmaq-audit-recos.md`](fedmaq-audit-recos.md)
(algorithm/math/paper-fidelity). This audit reconciles against those two — where a
concern is already covered there, it is demoted to a cross-reference, not re-raised
as new.
**Policy:** report-only (respects explore/confirm freeze). No code edited.

## Scope

Mechanism paths only: `strategy_hooks/fedmaq.py`, `kd_utils.py`,
`kd_loss_hook.py`, `client_hooks/standard.py`, `models.py` (`MobileNetV2GN` +
factories), `conf/algorithm/fedmaq.yaml`.

**Yardstick:** `.claude/rules/{flower-patterns,hydra-config,baselines}.md` +
general FL norms + Flower official strategy example (`FedCustom`, ctx7
`/flwrlabs/flower`).

**Not flagged (prior decisions):** MobileNetV2GN-as-default = DECISIONS.md
Decision 1; untuned mechanism hyperparameters = tracked in
`docs/plans/formal-experiment-plan.md` §2.

## Already covered by the design audit — cross-references only

These surfaced while reading but are **not new findings**; the design audit
examined them and ruled them defensible. Recorded here so the two audits agree.

- **Server-side gradient-norm computation** (`fedmaq.py:186-223`). Covered by
  `fedmaq-audit.md` §1.4 (⚠️ Low, "subtle design choice"). It answers both angles
  I would have raised: *why server-side vs client-reported* (preserves uplink
  budget; server already holds partition indices) and *single-batch noise*
  (Priority 3 EMA smoothing, §2.3). Because `g_k` is computed on the **global**
  model, server-side vs client-reported yields the identical deterministic number
  — there is **no output-affecting defect** in current runs. The only genuinely
  code-lens residue is a Flower-idiom note, logged as F1 below at its true (low)
  severity — not a correctness issue.
- **`_grad_norm_ema` unbounded growth** (`fedmaq.py:137`). Covered by
  `fedmaq-audit-recos.md` §10.4 (no action at thesis scale). My F5b/F7 concern a
  *different* dict (`_round_client_q`) — see F7.
- **Config default hygiene.** `fedmaq-audit-recos.md` §10.1 flags `ema_decay: 0.99`
  as a bad *value*. F8 below is the sibling *mechanism* issue (code fallback ≠
  shipped yaml), a different key set.

## New findings (in-scope code quality)

Severity: 🟠 medium · 🟡 low. No 🔴 — nothing here produces wrong output today.

### 🟠 F6 — Broad `except Exception` masks failures in 3 hot paths [robustness]

`fedmaq.py:216` (grad-norm → default 1e-8), `kd_utils.py:160` (teacher load →
silently drop teacher), `kd_utils.py:186` (whole server-KD → return un-distilled
params). A systematic fault (e.g. an arch/shape mismatch — exactly what
`set_model_parameters`' strict checks are designed to raise) is caught and
converted into a silently degraded ensemble or a skipped KD step, invisible across
a 40-round sweep. This is the audit's meatiest in-scope item: the design audit
never examines error handling.

**Failure scenario:** every teacher fails to load; the run continues on the raw
FedAvg aggregate with no KD and no error, and no metric records that KD was
skipped.

**Recommendation:** narrow the exception types, and emit a **dropped-teacher /
KD-skipped count as a WandB metric** (per `evaluation-metrics.md`) so silent
degradation is observable in run telemetry.

### 🟠 F2 — Grad-norm probe backprop is unaccounted in the server time model [efficiency · telemetry]

`fedmaq.py:202-223` runs a full MobileNetV2GN forward+backward on the server per
sampled client every round (plus a fresh DataLoader per client per round) to
extract one scalar. `server_sim_time` (`fedmaq.py:377-394` → `kd_server_sim_time`)
models **only** the KD passes (`num_public·kd_epochs·num_teachers`); the
per-client grad-norm backprop cost is **not** in the simulated server time. Since
wall-clock/compute is a tracked metric (`evaluation-metrics.md`), the server-side
cost of the adaptive-quantization signal is under-reported.

**Recommendation:** add the grad-norm probe cost to the server time model, or
report the norm from the client's existing fit backprop (removes the server cost
entirely; mirrors how DAdaQuant reports its plateau loss client-side).

### 🟡 F7 — `_round_client_q` never reset between rounds [hygiene]

`fedmaq.py:135,271`. Written per client in `configure_fit`, read in `aggregate_fit`
soft-voting (309). Never cleared — grows across the run and retains stale q for
clients not sampled this round. Safe **today** (aggregate_fit only reads cids the
same round's configure_fit refreshed) but fragile. Sibling to
`fedmaq-audit-recos.md` §10.4 (`_grad_norm_ema` growth).

**Recommendation:** clear at the top of `configure_fit`, or key by
`(server_round, cid)`.

### 🟡 F8 — Code defaults disagree with shipped yaml [config · convention]

`fedmaq.py:148,150` default `q_min=2`, `c_unit=2048.0`; `fedmaq.yaml:3,13` ship
`q_min=1`, `c_unit=512.0`. The `.get(key, default)` fallbacks silently substitute
a *different* value if a key is renamed/removed — no error. Per `hydra-config.md`
(hyperparameters live in config, not Python), the algorithm-defining knobs should
fail-loud on absence. Sibling to `fedmaq-audit-recos.md` §10.1.

**Recommendation:** for `q_min`/`q_max`/`c_unit`/`formulation`, read without a
default (or a raising sentinel) so a missing key is caught, not masked.

### 🟡 F4 — Stale ResNet18 reference in `FedMAQHook` docstring [doc-drift]

`fedmaq.py:128-130`: grad-norm model "avoiding repeated ResNet18 allocation."
Model is now MobileNetV2GN (or the KD student for `fedmaq_lite`). (Note: the design
audit §1.4 prose is itself slightly stale on which probe model is used — worth a
sync there too, out of this audit's edit scope.)

**Recommendation:** reword architecture-agnostic ("avoiding repeated model
allocation").

### 🟡 F5 — Instance attributes set outside `__init__` [hygiene]

`fedmaq.py:238,245,314` set `_last_grad_norms`, `_last_assigned_q`,
`_last_round_kd_metrics` lazily; `get_eval_metrics` guards each with `hasattr`.
`__init__` (135-137) already initializes three sibling fields — be consistent.

**Recommendation:** initialize all reported-state fields in `__init__`; drop the
`hasattr` dance.

### 🟡 F1 — Heavy compute inside `configure_fit` diverges from Flower idiom [convention]

`fedmaq.py:146-223`. Flower's `configure_fit` is conventionally lightweight
(sampling + config assembly, per the official `FedCustom` example); FedMAQ
front-loads model instantiation + per-client data loading + backprop there. This
is the *code-convention* residue of the design-audit-covered §1.4 choice — noted
for idiom, **not** a correctness defect (see cross-reference section).

**Recommendation:** none required for the thesis; if ever refactored, the
grad-norm probe is the natural thing to move out of `configure_fit`.

### 🟡 F9 — Per-round full-model deepcopy in client KD hook [efficiency · conditional]

`kd_loss_hook.py:56`. `on_train_begin` `copy.deepcopy`s the whole client model
every round; active only when `client_kd_reg=true` (off by default), so no impact
on the current grid — noted for when the KD-reg sweep runs. The design audit does
not examine `kd_loss_hook.py`.

**Recommendation:** if the reg sweep becomes hot, snapshot once and reload weights.

## Positives (no action — noted to prevent regressions)

- `set_model_parameters` (`models.py:300-319`) strict length + per-tensor shape
  checks — fails loud on arch mismatch. This is precisely what F6's broad excepts
  then re-hide; the guard is only useful if the exception isn't swallowed.
- `strict=True` on every param/delta `zip`.
- `configure_fit` re-instantiates `FitIns` (fedmaq.py:269) to avoid shared-ref
  overwrite, with an explanatory comment.
- Two-tier `min()`-then-single-floor-into-Q is well-documented (fedmaq.py:114-120).
- `FedMAQFit` is an intentional 3-line `StandardFit` subclass (standard.py:158) —
  the handoff's "missing dedicated fedmaq client hook" concern is resolved, not a
  gap.

## Summary table

| ID | Severity | Category | File:line | Status vs design audit |
|----|----------|----------|-----------|------------------------|
| F6 | 🟠 | robustness | fedmaq.py:216, kd_utils.py:160,186 | new |
| F2 | 🟠 | efficiency · telemetry | fedmaq.py:202,377 | new (extends §5.4) |
| F7 | 🟡 | hygiene | fedmaq.py:135,271 | new (sibling §10.4) |
| F8 | 🟡 | config | fedmaq.py:148,150 | new (sibling §10.1) |
| F4 | 🟡 | doc-drift | fedmaq.py:128 | new |
| F5 | 🟡 | hygiene | fedmaq.py:238 | new |
| F1 | 🟡 | convention | fedmaq.py:146 | cross-ref §1.4 |
| F9 | 🟡 | efficiency · conditional | kd_loss_hook.py:56 | new |
| — | — | — | fedmaq.py:186 (server-side g_k) | **covered §1.4, no new finding** |

**Headline:** mechanism logic is clean, strict-checked, and well-commented. The
design audit already cleared the algorithm-design questions; this code-lens pass
adds two items with teeth — **F6** (silent failure-masking hides exactly the arch
mismatches the strict param checks exist to catch) and **F2** (server-side
grad-norm cost missing from the tracked runtime) — plus low-severity config/hygiene
cleanups. Nothing here is output-wrong today, consistent with the explore/confirm
freeze.
