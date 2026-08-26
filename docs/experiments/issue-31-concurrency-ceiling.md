# Per-Host Throughput and Concurrency Ceiling (Issue #31)

**Status:** telemetry re-derived and verified; measurement commands prepared; awaiting
user-supplied results. Not an ADR — no decision has been made yet, only the
groundwork for one.

**Where this lives:** `docs/experiments/` registers "exploratory sweeps that carry
hand-written analysis" as a `results.md`/`comments.md` pair
([`docs/experiments/README.md`](README.md)). This ticket has no results yet — that
pair convention doesn't fit a pre-measurement note. No other `docs/` location fits
better: `docs/audits/` is closed-out foundational audits that fold into ADRs and get
deleted, and this isn't a durable decision yet, so it isn't an ADR either. This is a
single file by design, placed here because it is the closest existing shelf. Once
the three runs in Part B return results, either extend this file with a `## Results`
section or split it into the registry's normal `results.md`/`comments.md` pair —
whichever the person closing the ticket prefers. **The result numbers themselves
belong in the GitHub issue** (`gh issue comment 31 ...`), per AGENTS.md's "GitHub
Issues are the sole live-state record" — this file carries method and reusable
formulas, not run state.

---

## Part A — What the existing telemetry already settles

Re-derived directly from `fedmaq-analyses/data/hydra_outputs/v1-provisional/`
(177 run directories) against `fedmaq-analyses/src/fedmaq_analysis/loaders.py` and
`fedmaq-experiments/src/fedmaq/core/telemetry.py`, not quoted from the ticket.
Three of the 177 runs (`benchmark_grid__cifar10__{feddistill,fedkd}__default__a1.0__fnone__s0`,
`benchmark_grid__cifar10__fedmaq__default__a1.0__f2__s0`) have no `experiment_log.csv`;
their manifest (`data/manifests/*.yaml`) points `telemetry_path` at
`experiment_log.repaired.csv` (`integrity_status: derived_exact_dedupe`) — this is
the analysis pipeline's own fallback, not an ad hoc choice made for this ticket, and
all totals below include those three via that file.

### The headline numbers confirm, almost to the decimal

| Claim | Ticket | Re-derived | Verdict |
| --- | --- | --- | --- |
| Total wall time, 177 cells | ~109 h | **109.07 h** | confirmed |
| Median run wall time | 0.57 h | **0.57 h** | confirmed exactly |
| Max run wall time | 1.58 h | **1.58 h** (`benchmark_grid__cifar10__fedmaq__default__a0.1__f2__s42`) | confirmed exactly |
| `system/cumulative_time_sec` sum | 1055 h | **1055.07 h** | confirmed |
| Ratio of the two | ~10x | **9.67x** | confirmed as a ratio — **mislabeled**, see below |
| Client-side share of compute | ~98–100% every algorithm | 96.6–100% depending on scope (see table) | confirmed, with the FedMAQ figure nearer 97% than 98% |
| Server-side share | 0% for six of seven, ~2.3% for FedMAQ | **0.000% for six baselines, 2.29% for FedMAQ** (81 FedMAQ-family runs) | confirmed almost exactly |
| FedKD cost outlier | ~25% above the rest | **+19% to +26%** across matched (dataset, α, seed) triples, mean ≈ +22% on CIFAR-10/100; **inverted on FEMNIST** | confirmed for CIFAR, exception found on FEMNIST |

### Correction: the "~10x concurrency" line is a category error, not a rounding issue

`system/cumulative_time_sec` is **not** GPU-actor compute time. Its components are
computed at `src/fedmaq/core/telemetry.py:255-259` and
`src/fedmaq/core/strategy.py:242-259`:

```python
client_sim_time = max(round_delays)   # max over the round's sampled clients
server_sim_time = strategy.hook.server_sim_time(...)
round_time = client_sim_time + server_sim_time
```

`round_delays` comes from `strategy.cost_model.client_round_delay(...)`, which is
driven entirely by `bandwidth_mbps` and `compute_samples_per_sec` — the **synthetic
Raspberry Pi 5 / 802.11ac cost model** derived in
[ADR-0002](../adr/0002-hardware-telemetry-grounding.md) (20.0 samples/sec sustained
CPU throughput, 10 Mbps WiFi). It answers "how long would this round take on real,
distributed edge hardware," and accumulates that projection across rounds and cells.

`system/wall_time_sec` is a different quantity entirely — `time.perf_counter()`
deltas around the actual round on the shared A100 host
(`telemetry.py:273-275`). Its cumulative sum (109 h) is the real infrastructure cost
already paid.

**The 9.67x ratio is therefore (simulated-Pi5-fleet-deployment-time) /
(actual-GPU-simulation-time), a byproduct of "a datacenter GPU is much faster than a
Raspberry Pi 5 over WiFi" — not a measurement of how many Ray client actors ran
concurrently on the host.** It says nothing about per-host concurrency. **Do not
carry the 9.67x (or "~10x") figure forward into #22, #29, or the manuscript as a
concurrency measurement — it isn't one.** It is the *only* evidence the ticket
offers that "concurrency is already being extracted," and that evidence doesn't
hold up; without it the claim is unsupported, not just under-sourced. The
client/server split (next table) is a separate, independently valid finding that
does not depend on this ratio, and it is what actually carries the "no
server-side bottleneck" conclusion. This correction doesn't weaken the ticket's
ask; if anything it sharpens it: **there is zero existing signal on the actual
concurrency question**, because every one of the 177 v1-provisional
runs used `experiment.client_gpus=0.5` — confirmed by direct read of every
`run_manifest.json`, no exceptions. The only paired data point that exists at all is
[ADR-0013](../adr/0013-execution-infrastructure-failures.md)'s own probe: two 5-round
FedMAQ runs, back to back, 0.5 vs 1.0, which found 0.5 *faster* (~5.0 GB VRAM at 0.5
vs ~3.0 GB at 1.0 of a 40 GB card, 24 GB free) — but 0.25 and 0.125 have never been
tried. That gap is exactly what Part B measures.

The client-side/server-side split, by contrast, **is** a valid and useful finding on
its own terms — both halves are the same synthetic-cost-model unit, so their ratio to
each other is meaningful even though their ratio to wall time is not. It supports the
ticket's conclusion ("no server-side or KD bottleneck; speedup comes from client
concurrency") for an even more direct reason than the ticket states: the server side
of the *deployment cost model* is negligible, so nothing about a faster server would
move the deployment-time projection, and nothing about the existing telemetry
contradicts pursuing client-side concurrency — it simply never measured it.

### Per-algorithm compute split (`system/cumulative_client_time_sec` vs `..._server_time_sec`)

| Scope | n | Client % | Server % |
| --- | --- | --- | --- |
| dadaquant (benchmark_grid) | 15 | 100.000% | 0.000% |
| fedavg (benchmark_grid) | 15 | 100.000% | 0.000% |
| feddistill (benchmark_grid) | 15 | 100.000% | 0.000% |
| fedkd (benchmark_grid) | 15 | 100.000% | 0.000% |
| fedpaq (benchmark_grid) | 15 | 100.000% | 0.000% |
| fedprox (benchmark_grid) | 15 | 100.000% | 0.000% |
| fedmaq (benchmark_grid only, n=15) | 15 | 96.556% | 3.444% |
| **fedmaq (all FedMAQ-family cells: benchmark_grid + ablation + formulation_study + uniform_memory_control, n=81)** | 81 | **97.71%** | **2.29%** |
| fedavg_kd (ablation-only arm, not one of the seven baselines) | 6 | 97.77% | 2.23% |
| **All 177 cells** | 177 | 98.69% | 1.31% |

`fedavg_kd` is an ablation control arm (dual-model KD on top of FedAvg), not one of
the seven baseline-stack algorithms in `.agents/rules/experiment-design.md`'s table —
its nonzero server share doesn't contradict "six of seven."

### FedKD's outlier, checked properly (matched conditions, not a mixed-condition mean)

A naive mean across all 15 `fedkd` cells vs. the other six algorithms' 15-each mean
understates the effect (+6% only) because it silently averages in FEMNIST, whose
`total_rounds`/config differ enough to invert the comparison. The correct check
holds `(dataset, alpha, seed)` fixed and compares FedKD to the mean of the other six
algorithms in that exact cell:

| Dataset | α | FedKD vs. other-six mean |
| --- | --- | --- |
| cifar10 | 0.1 | +25.2% to +26.0% (3 seeds) |
| cifar10 | 1.0 | +21.7% to +22.3% |
| cifar100 | 0.1 | +21.1% to +22.2% |
| cifar100 | 1.0 | +19.3% to +19.4% |
| femnist | 1.0 | **−66%** (FedKD is *cheaper*, not costlier) |

CIFAR-10/100 (12 of 15 cells) average ≈ +22%, consistent with the ticket's "~25%"
and with the dual-model-training rationale in `experiment-design.md`. **FEMNIST is a
genuine exception**, not a rounding artifact — worth a one-line flag if this number
is ever cited outside this file, but it doesn't change the six-vs-seven verdict since
FEMNIST is 3 of FedKD's 15 cells and the CIFAR majority carries the claim.

### Wall-time baseline, at the only setting ever run (`client_gpus=0.5`)

Needed as the calibration baseline for Part C's dispatch-plan formula — captured here
so it isn't re-derived twice.

| Stage (per #22) | n cells | Sum wall h | Mean h/cell |
| --- | --- | --- | --- |
| Stage 1 — `formulation_study` | 30 | 23.03 | 0.768 |
| Stage 2 — `benchmark_grid`(×3) + `ablation` + `uniform_memory_control` | 147 | 86.04 | 0.585 |
| **All 177 (v1-provisional)** | 177 | 109.07 | 0.616 |

By dataset: cifar10 mean 0.683 h/cell (n=114), cifar100 mean 0.589 h/cell (n=42),
femnist mean 0.311 h/cell (n=21) — femnist cells are cheap; weight the projection in
Part C accordingly if the rerun's dataset mix differs from v1's.

---

## Part B — Paste-ready JupyterHub commands

**Per AGENTS.md, no run in this section has been executed by this agent.** Paste
these into the JupyterHub terminal on the datacenter allocation and report back the
requested fields (as a comment on issue #31, or directly in this session).

### Step 1 — Four concurrency probes: three settings plus a same-session repeat control

ADR-0013 measured `client_gpus` 0.5 vs 1.0 with "two 5-round FedMAQ runs, back to
back in one session so the shared host's co-tenant load could not drift between
them, differing only in `client_gpus`." Two things are strengthened here relative to
a literal extension of that methodology, both because the stakes are a three-way
comparison rather than a pass/fail check:

- **5 rounds is too short at this dataset/α.** The existing v1-provisional telemetry
  (Part A) shows the *same* FedMAQ/cifar10/α=0.1 config took 0.73 h, 1.14 h, and
  1.58 h across its three seeds for 100 rounds on this shared host — a 2.2x spread
  from co-tenant drift alone. A 5-round probe (~170–360 s) is mostly warmup and
  noise; **use 15 rounds** (roughly 8–20 min per run, still cheap) so the
  steady-state per-round rate in Part C averages over enough rounds to be readable
  against that noise floor.
- **A three-point comparison needs a noise control, not just back-to-back
  ordering.** Run `0.5 → 0.25 → 0.125 → 0.5` (repeat) in one sitting. If the two
  `0.5` runs disagree by more than the gaps between settings, the co-tenant host is
  moving faster than the effect being measured and **that itself is the finding** —
  report it as "noise-dominated, no ceiling determinable from this pass" rather than
  picking a winner from numbers that don't mean what they'd otherwise mean.

Same algorithm, dataset, α, seed throughout — only `client_gpus` changes.
`algorithm.post_process` is left at its config default; it's irrelevant to a
timing-only probe (ADR-0013 established `client_gpus` doesn't move outcomes). The
`time` prefix gives a robust `real` wall-clock reading independent of the CSV, in
case a run is interrupted before its telemetry file is flushed.

**Expected outcomes, stated up front so a flat result doesn't read as a broken
probe:** `client_gpus` caps how many client actors *may* run at once
(`floor(1/G)`: 2 at 0.5, 4 at 0.25, 8 at 0.125) against 10 clients sampled per round
(`num_clients=100 × client_fraction=0.1`) — it does not guarantee the GPU has
headroom to run them concurrently. If two concurrent MobileNetV2GN client actors
already saturate the A100's SM occupancy, 4 and 8 actors will simply time-slice the
same silicon and wall time will barely move between 0.25 and 0.125. **That is a
valid ceiling, not a failed measurement** — it means the ceiling sits at or below
0.5, and #22/#29's dispatch plan scales by adding hosts (N), not by dropping
`client_gpus` further on one host.

```bash
mkdir -p outputs/diagnostic/issue-31-concurrency-probe

time uv run python scripts/run.py \
  dataset=cifar10 heterogeneity=dirichlet_alpha_0.1 algorithm=fedmaq \
  experiment.total_rounds=15 seed=0 experiment.client_gpus=0.5 \
  ray.temp_dir=/tmp/ray-cjb ray.object_store_gb=4 \
  hydra.run.dir=outputs/diagnostic/issue-31-concurrency-probe/client_gpus_0.5_a \
  2>&1 | tee outputs/diagnostic/issue-31-concurrency-probe/client_gpus_0.5_a.log

time uv run python scripts/run.py \
  dataset=cifar10 heterogeneity=dirichlet_alpha_0.1 algorithm=fedmaq \
  experiment.total_rounds=15 seed=0 experiment.client_gpus=0.25 \
  ray.temp_dir=/tmp/ray-cjb ray.object_store_gb=4 \
  hydra.run.dir=outputs/diagnostic/issue-31-concurrency-probe/client_gpus_0.25 \
  2>&1 | tee outputs/diagnostic/issue-31-concurrency-probe/client_gpus_0.25.log

time uv run python scripts/run.py \
  dataset=cifar10 heterogeneity=dirichlet_alpha_0.1 algorithm=fedmaq \
  experiment.total_rounds=15 seed=0 experiment.client_gpus=0.125 \
  ray.temp_dir=/tmp/ray-cjb ray.object_store_gb=4 \
  hydra.run.dir=outputs/diagnostic/issue-31-concurrency-probe/client_gpus_0.125 \
  2>&1 | tee outputs/diagnostic/issue-31-concurrency-probe/client_gpus_0.125.log

# Repeat control — same setting as the first run, run last. Compares against
# client_gpus_0.5_a to separate "the setting changed the speed" from "the host did."
time uv run python scripts/run.py \
  dataset=cifar10 heterogeneity=dirichlet_alpha_0.1 algorithm=fedmaq \
  experiment.total_rounds=15 seed=0 experiment.client_gpus=0.5 \
  ray.temp_dir=/tmp/ray-cjb ray.object_store_gb=4 \
  hydra.run.dir=outputs/diagnostic/issue-31-concurrency-probe/client_gpus_0.5_b \
  2>&1 | tee outputs/diagnostic/issue-31-concurrency-probe/client_gpus_0.5_b.log
```

If any single command looks hung well past ~30 minutes, that is itself a data point
— `Ctrl+C`, note it, and report it rather than waiting indefinitely. ADR-0013's
diagnosis: a hang here is either the low-system-RAM deadlock this ticket is checking
for, or the unrelated SuperNode-membership defect it already fixed with retries —
either is worth reporting, but don't let it block the remaining probes.

**Paste back, per run:**
1. The `real` line from `time`.
2. The **full** `system/wall_time_sec` (or `system/cumulative_wall_time_sec`) column
   from `outputs/diagnostic/issue-31-concurrency-probe/client_gpus_<label>/experiment_log.csv`
   — all 15 rounds, not just the endpoints. It's ~15 numbers per run already sitting
   in the open CSV, and it's what lets Part C check whether the per-round rate is
   stable (a flat, low-variance column) or drifting (co-tenant contention mid-run) —
   two endpoints alone can't distinguish those.
3. Any Ray/CUDA warnings in the `.log` file (object store spill warnings, OOM-killer lines, retry/backoff messages from the partition-ID fix).
4. `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader` run once during each probe, for a cheap cross-check against ADR-0013's VRAM numbers (expected: still far from the 40 GB ceiling at all four runs, per ADR-0013's ~24 GB-free finding at 0.5).

### Step 2 — RAM and Ray object-store pressure at the setting that wins Step 1

This is conditional on Step 1's outcome, so it can't be fully pre-written — fill in
`<WINNER>` with whichever of 0.5/0.25/0.125 Step 1 shows as fastest (or, if Step 1
comes back noise-dominated or flat per the "expected outcomes" note above, use 0.5 —
there's no basis to run more concurrently than the point already shown safe). Run
this **longer** than the Step 1 probes — RAM/object-store pressure and the
documented deadlock mode are the kind of thing that builds up over a run, and
ADR-0013 notes explicitly that 5 rounds "cannot" surface a mid-run failure. 20 rounds
is enough to see a trend without spending a full 100-round budget on a diagnostic;
extend to 100 if the trend looks borderline.

**What the deadlock looks like in the sampler log:** the documented failure is Ray
waiting on an actor that never starts, not a slow RAM leak — so the log's signature
is `avail_MB` sitting flat (not necessarily near zero) while the run's own stdout
stops advancing past a round. A flat sampler log paired with a stalled run *is* the
positive signal here, not an absence of one — don't wait for `avail_MB` to visibly
drop before treating a stall as the failure mode in question.

Start a background sampler *before* the run, in a separate terminal/cell so it keeps
writing even if the run itself hangs — that failure case is exactly what this is
watching for:

```bash
mkdir -p outputs/diagnostic/issue-31-concurrency-probe

( while true; do
    date +%T
    free -m | awk '/Mem:/{print "RAM total_MB="$2" used_MB="$3" avail_MB="$7}'
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "---"
    sleep 15
  done ) > outputs/diagnostic/issue-31-concurrency-probe/ram_vram_samples.log 2>&1 &
echo "sampler pid: $!"

# Best-effort — Ray's object-store CLI surface has changed across versions (this
# host runs ray==2.55.1 per uv.lock). Try one of these; if both error, skip them —
# free -m above is what actually answers the ticket's stated concern (system RAM,
# not the object store specifically), so this is a bonus, not a blocker.
ray status 2>&1 | tee -a outputs/diagnostic/issue-31-concurrency-probe/ram_vram_samples.log || true
```

Then, in the foreground:

```bash
time uv run python scripts/run.py \
  dataset=cifar10 heterogeneity=dirichlet_alpha_0.1 algorithm=fedmaq \
  experiment.total_rounds=20 seed=0 experiment.client_gpus=<WINNER> \
  ray.temp_dir=/tmp/ray-cjb ray.object_store_gb=4 \
  hydra.run.dir=outputs/diagnostic/issue-31-concurrency-probe/sustained_client_gpus_<WINNER> \
  2>&1 | tee outputs/diagnostic/issue-31-concurrency-probe/sustained_client_gpus_<WINNER>.log

# stop the sampler once the run finishes (use the pid the sampler command printed —
# it was backgrounded in a separate shell/cell, so a bare `kill %1` here targets the
# wrong job table)
kill <sampler pid printed above>
```

**Paste back:**
1. `ram_vram_samples.log` in full (or its `RAM avail_MB` min/max if the full log is long).
2. Whether `avail_MB` trended down over the run, stayed flat, or oscillated.
3. Whether the run completed cleanly, and the `real` wall time.
4. Peak `nvidia-smi` VRAM used (ours) — sanity check against ADR-0013, which found VRAM was never close to the ceiling; if this run shows otherwise at higher concurrency, that's a materially new finding.

---

## Part C — Analysis framework

This is written to be filled in mechanically once Part B's numbers come back — no
new judgment calls should be needed, only arithmetic. If a step's assumption turns
out to be wrong (e.g. throughput doesn't scale the way step 3 assumes), that's worth
flagging explicitly rather than silently patched over.

### 0. Noise check — does the repeat control invalidate the comparison?

Before anything else, compare `client_gpus_0.5_a` and `client_gpus_0.5_b` (Step 1's
repeat control). Compute each run's steady-state per-round rate (step 1 below) and
take the difference between the two `0.5` runs as `noise_floor_sec`. If
`noise_floor_sec` is comparable to or larger than the *gap* between the 0.5 and
0.125 rates, the three-way comparison is noise-dominated — report that plainly
("no ceiling determinable from this pass; co-tenant drift exceeded the measured
effect") rather than picking whichever ran fastest as if it meant something.
Otherwise, average the two `0.5` runs and proceed.

### 1. Strip round-1 warmup, get steady-state per-round wall time

For each run (using all 15 rounds' `system/wall_time_sec` pasted back, not just the
endpoints — this also lets a visibly non-flat column be caught here rather than
silently averaged away):

```
per_round_wall_sec(G) = mean(wall_time_sec[round=2..15])   # exclude round 1's one-time warmup
```

Report the standard deviation alongside the mean; a high spread relative to the
between-setting gaps is the same noise-dominated signal as step 0, just visible
within one run instead of across the repeat pair.

### 2. Project to a full 100-round cell, at each `client_gpus` setting

```
projected_100round_wall_sec(G) = wall_at_round_1(G) + 99 * per_round_wall_sec(G)
```

### 3. Throughput and speedup factor, relative to the 0.5 baseline already measured across 177 cells

```
throughput_cells_per_hour(G) = 3600 / projected_100round_wall_sec(G)
speedup_factor(G) = projected_100round_wall_sec(0.5) / projected_100round_wall_sec(G)
```

`speedup_factor(0.5) = 1` by construction — it's the anchor, not a fourth data point.
If ADR-0013's own 0.5-vs-1.0 pair is wanted as a fourth anchor for a monotonicity
check, its 5-round runs would need the same round-1-stripping treatment; it wasn't
logged with that in mind, so treat it as directional confirmation ("more concurrency
helped, once") rather than a plug-in number.

### 4. Concurrency ceiling — decision rule

Tabulate `throughput_cells_per_hour(G)` for G = 0.5, 0.25, 0.125 (and 1.0 from
ADR-0013 as a sanity anchor, non-quantitative). Smaller `client_gpus` means more
client actors can share the GPU concurrently (`floor(1/G)`: 2 at 0.5, 4 at 0.25, 8 at
0.125), so throughput should rise as G shrinks *until something else saturates*.

- **If throughput keeps rising at 0.125** (i.e. `throughput(0.125) > throughput(0.25) > throughput(0.5)`
  with no sign of flattening): the ceiling wasn't reached in this test. Don't block
  #22/#29 on finding the exact asymptote — adopt 0.125 as the dispatch setting (it's
  the best measured point) and note in the issue that the true ceiling is somewhere
  beyond it, in case a future session wants to test 0.0625.
- **If throughput flattens or drops between 0.25 and 0.125**: the ceiling is at 0.25
  (or wherever the flattening starts).
- **Hard override — RAM safety floor, independent of throughput**: if Part B Step 2's
  sampler shows `avail_MB` trending toward zero (or dropping below roughly 10–15% of
  the host's 64 GB, i.e. under ~6.5–10 GB free) at the throughput-preferred `G`, the
  ceiling is capped at the next-larger `G` (less concurrency) that keeps headroom
  comfortable, regardless of what the throughput numbers alone would suggest. This
  is the documented deadlock mode (`--run_timeout_seconds` help text,
  `scripts/run_matrix.py:156-161`: "Ray can deadlock waiting on an actor that never
  starts, which is the documented low-system-RAM failure mode") — a hang here is not
  recoverable by throughput, it stalls the whole sweep.

```
G_ceiling = min(G_throughput_optimal, G_ram_safe)   # "min" in the sense of "less concurrent", i.e. the larger client_gpus value if they disagree
measured_per_host_throughput = throughput_cells_per_hour(G_ceiling)
```

### 5. Dispatch plan — measured per-host throughput × N

Using Part A's baseline table (all measured at `client_gpus=0.5`) and the
`speedup_factor(G_ceiling)` from step 3:

```
projected_stage_wall_sec(stage, G_ceiling) = stage_wall_sec_at_0.5 / speedup_factor(G_ceiling)

Stage 1 (formulation_study, 30 cells): stage_wall_sec_at_0.5 = 23.03 h
Stage 2 (147 cells):                    stage_wall_sec_at_0.5 = 86.04 h
```

This assumes the concurrency speedup measured on one cell type (FedMAQ/cifar10/α0.1)
generalizes across the other cell types in the grid (other algorithms, cifar100,
femnist). That's a real assumption, not a certainty — communication-vs-compute
balance differs somewhat by algorithm (FedKD's dual-model cost, femnist's much
smaller model). It's good enough to unblock #22/#29's planning; if tighter precision
matters later, repeat one probe pair on a FedKD cell and a femnist cell and compare
speedup factors before finalizing. Also note #22 reruns under a corrected
quantizer/byte-accounting (#24/#25) — that changes what's transmitted, not the
training compute this measurement is calibrating, so the baseline table should still
be a reasonable planning input, but flag it as v1-provisional-derived if precision is
later questioned.

Then, per #29's `--shard i/N`:

```
stage_wall_clock_with_N_hosts = projected_stage_wall_sec(stage, G_ceiling) / N
total_wall_clock ≈ stage1_wall_clock_with_N_hosts + stage2_wall_clock_with_N_hosts
```

(Sequential across stages — #22's Stage 2 is configured by Stage 1's freeze — but
each stage's own cells shard independently across the N hosts.) Solve for N given a
target completion window, or for the completion window given however many hosts the
adviser provisions — whichever is the free variable when this is filled in; this
framework doesn't presume one over the other. **This is the number #22's dispatch
plan and #29's N should be built from, once G_ceiling and the two probe wall-times
are in hand.**
