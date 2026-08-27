# Concurrency mechanism and throughput-ceiling status (issue #31)

Date: 2026-08-27. Background research for
[FedMAQ/fedmaq-experiments#31](https://github.com/FedMAQ/fedmaq-experiments/issues/31)
("Measure per-host throughput and concurrency ceiling"), part of the wayfinder
map for #21. No file outside this document was modified, no run was dispatched,
no configuration was changed — consistent with `AGENTS.md`'s "Agents do not run
experiments."

**Location note.** `docs/experiments/issue-31-concurrency-ceiling.md` already
exists at the issue's own linked path and is the operational groundwork
document (paste-ready probe commands, analysis formulas, dispatch-plan
framework) — it is explicitly a live handoff artifact awaiting user-run
results, not a closed research note, so this document does not edit or
supersede it. `docs/research/` is the directory this repo already uses for a
standalone investigative pass tied to one issue (precedent:
`docs/research/2026-08-27-soft-quality-formulation-grounding.md`, filed for
issue #32 for the same reason: `docs/adr/` is decisions, `docs/audits/` is
closed code-fidelity audits, `docs/experiments/` is run output/handoff). This
note is filed there, dated like that precedent.

**Scope.** This note independently derives the measured per-host throughput
baseline straight from raw telemetry (not from either doc's summary table),
re-verifies (against the actual source files, not the prose of either the
ticket or the existing groundwork doc) the mechanism behind the "~10x
concurrency" figure and the real per-host concurrency mechanism, confirms the
current measurement status (has anything actually been run since the
groundwork doc was written?), and reasons from code/config about the likely
bottleneck to the extent it is observable without new runs — per the ticket,
host VRAM/RAM live telemetry is out of scope to source directly.

---

## Bottom line

Two numbers answer the ticket's two halves differently.

**What one instance already yields, at the only setting ever run
(`client_gpus=0.5`, 2 concurrent client actors/round): 1.62 cells/hour**
(177 cells / 109.07 h), independently aggregated in this pass directly from
every `experiment_log.csv` / `experiment_log.repaired.csv` in
`fedmaq-analyses/data/hydra_outputs/v1-provisional/` — not quoted from the
ticket or from `docs/experiments/issue-31-concurrency-ceiling.md` (see
Section 1). This is a real, measured per-host throughput baseline, and it is
usable directly in #22/#29's dispatch-plan arithmetic today.

**Where adding concurrency stops paying is still unknown.** No probe output
exists anywhere in this repo or the sibling `fedmaq-analyses` checkout
(`outputs/diagnostic/issue-31-concurrency-probe/` does not exist; verified by
filesystem search — see "Measurement status" below). The only empirical data
point at any `client_gpus` value other than 0.5 is ADR-0013's own two 5-round
FedAvg runs (0.5 vs. 1.0), which is too short to strip warmup and was run on
FedAvg, not FedMAQ. The concurrency mechanism itself is fully pinned down from
source (Flower's own installed package, not just this repo's wrapper), and is
summarized below — that part of the ticket's Question 2 has a complete,
code-grounded answer even though the ceiling itself does not.

---

## 1. Measured per-host throughput at the only setting ever run

Aggregated directly from the 177 `experiment_log.csv` (or, where absent,
`experiment_log.repaired.csv`) files under
`fedmaq-analyses/data/hydra_outputs/v1-provisional/` — the raw telemetry, not
the ticket's prose and not `docs/experiments/issue-31-concurrency-ceiling.md`'s
Part A table, per this task's "primary sources only" instruction. Aggregated
with an ad hoc script (not committed to this repo — a scratch analysis, run
with the repo's own `.venv/Scripts/python.exe`), summing `system/wall_time_sec`
per run directory
(and cross-checked against each run's final `system/cumulative_wall_time_sec`
row, per the accumulator relationship at `telemetry.py:366-369`):

```
n run dirs processed: 177
n using repaired fallback: 3
  -> benchmark_grid__cifar10__feddistill__default__a1.0__fnone__s0
  -> benchmark_grid__cifar10__fedkd__default__a1.0__fnone__s0
  -> benchmark_grid__cifar10__fedmaq__default__a1.0__f2__s0
n missing entirely: 0

Total wall-time, all 177 cells:        109.0707 h
  (sum-of-rounds method and final-cumulative-column method agree to 4 decimal
   places for all 177 cells — 0 discrepancies over 1%)
Median per-cell:                         0.5657 h
Max per-cell:                            1.5754 h
  (benchmark_grid__cifar10__fedmaq__default__a0.1__f2__s42)
Mean per-cell:                           0.6162 h

Throughput, single cell at a time, client_gpus=0.5, 2 actors/round:
  1.6228 cells/hour
```

This reproduces the ticket's headline 109h / 0.57h / 1.58h figures to within
rounding, independently and from the raw per-round telemetry rather than by
trusting either secondary source, and additionally derives the
**cells/hour throughput figure the ticket itself asks for** (Part C's
`throughput_cells_per_hour(G)` formula in the groundwork doc, evaluated at the
one `G` that has ever actually been run). Because Section 4a below establishes
that all 177 cells ran strictly sequentially — one `subprocess.run` at a time,
never overlapping — this 1.62 cells/h figure *is* the per-host throughput at
`client_gpus=0.5` with no cross-cell concurrency confound; it needs no
correction for cells running in parallel with each other.

**What this number does not answer:** whether 1.62 cells/h is the ceiling, or
whether `client_gpus=0.25`/`0.125` (4/8 actors per round instead of 2) would
raise it. That comparison requires runs at those settings, which do not exist
(next section).

---

## 2. Measurement status — has anything been run since the groundwork doc?

```
find . -iname "*concurrency*"          # only the .md files themselves
find . -iname "*diagnostic*"           # no outputs/diagnostic tree anywhere
```

`outputs/` in this checkout contains only `outputs/ci/` and `outputs/golden/`
(pre-existing golden-diff fixtures, unrelated to #31). `docs/agents/execution-model.md:29`
states plainly: "`outputs/` being empty locally proves nothing... Results live
on the [datacenter] allocation" — so this is not conclusive that no probe was
run on the JupyterHub host, only that no probe result has been pasted back
into either repo checked here. The issue's single comment (`gh issue view 31
--comments`) ends with "Groundwork done, awaiting your run results — leaving
this open," and there is no second comment. **Per the ticket's own framing,
the probes in Part B of the groundwork doc have not been executed.**

This means the ticket's central question — "how much throughput does one
instance already yield, and where does adding concurrency stop paying" — is
genuinely open, not just under-documented. Everything below narrows the
*mechanism* and the *candidate bottleneck*, not the ceiling number itself,
which requires the runs the groundwork doc already prepared commands for.

---

## 3. Where the ~10x figure actually comes from, and where it does not

Independently re-derived from `src/fedmaq/core/telemetry.py`, not quoted from
either doc.

- `system/cumulative_time_sec` accumulates `round_time = client_sim_time +
  server_sim_time` (`telemetry.py:291-295`), where `client_sim_time =
  max(round_delays)` and each delay comes from
  `strategy.cost_model.client_round_delay(...)` (`telemetry.py:275-284`) —
  driven by `bandwidth_mbps` and `compute_samples_per_sec`, the synthetic
  Raspberry Pi 5 / 802.11ac cost model set in `conf/experiment/default.yaml:20-21`
  (`compute_samples_per_sec: 20.0  # Raspberry Pi 5 Quad Cortex-A76 @ 2.4GHz
  sustained`) and derived in ADR-0002. This is a projected edge-fleet
  deployment time, not a host measurement.
- `system/wall_time_sec` is a `time.perf_counter()` delta taken once per round
  around the real Ray/Flower round (`telemetry.py:309-311`, `self._last_wall_ts`
  set at `telemetry.py:107`). This is the actual GPU-host time.
- Independently confirmed: all 177 `run_manifest.json` files in
  `fedmaq-analyses/data/hydra_outputs/v1-provisional/` report `"client_gpus":
  0.5` with zero exceptions —

  ```
  $ grep -oh '"client_gpus"[^,}]*' fedmaq-analyses/.../*/run_manifest.json | sort | uniq -c
      177 "client_gpus": 0.5
  ```

  — so **every existing wall-time data point in the v1-provisional bundle was
  produced at one single concurrency setting.** There is no existing wall-time
  comparison across `client_gpus` values to derive a concurrency ratio from in
  the first place, independent of what `cumulative_time_sec` even measures.

**Conclusion, independently reached:** the ~9.67x ratio (1055h / 109h) is
(simulated-Pi5-fleet-time)/(actual-host-wall-time) — a byproduct of the cost
model's constants, not a count of concurrently-running client actors. This
confirms the groundwork doc's correction; it is not a new finding, but it is
now verified against the telemetry code directly rather than accepted from the
doc's prose. **The client/server compute split (client ~97-99%, server
~0-2.3%) is a separate, valid finding — it is a ratio between two components
of the same synthetic cost model, so it holds regardless of the wall-time
question.**

---

## 4. The actual concurrency mechanism (Question 2, answered from source)

Two independent layers determine how much runs at once. Both are fully
resolved from source with no runs needed.

### 4a. Across the 177 cells: no concurrency at all

`scripts/run_matrix.py:344-364` drives the sweep with a plain `for idx, task in
enumerate(tasks, 1)` loop, calling `subprocess.run(task["cmd"], timeout=...)`
and blocking on each one before starting the next; `kill_ray_processes()` runs
before every task (`run_matrix.py:357`) specifically to guarantee a clean Ray
cluster per cell. There is no `Pool`, `ThreadPoolExecutor`, `Popen`-and-move-on,
or any other construct in the file that would let two matrix cells run
concurrently (`grep`-verified: no such symbol appears anywhere in
`run_matrix.py`). **The 177 cells ran one at a time, full stop.** Whatever
concurrency exists is entirely *within* one cell's federated rounds, not
across cells.

### 4b. Within one cell: Ray's fractional-GPU actor pool, sized by a formula in Flower itself

`src/fedmaq/simulation.py:320-325` sets:

```python
backend_config: dict[str, dict[str, ConfigRecordValues]] = {
    "client_resources": {
        "num_cpus": 1,
        "num_gpus": float(OmegaConf.select(cfg, "experiment.client_gpus", default=0.0)),
    }
}
```

and calls `flwr.simulation.run_simulation(..., num_supernodes=cfg.experiment.num_clients,
backend_config=backend_config)` (`simulation.py:333-338`). `num_supernodes` is
`conf/experiment/default.yaml:2`'s `num_clients: 100`; `client_fraction: 0.1`
(`default.yaml:9`) means 10 of those 100 are sampled per round — this is the
pool of *candidate* clients per round, not the number that run concurrently.

The number that actually run concurrently is computed by Flower's own
installed package (flwr 1.32.1 per `uv.lock`), and the live chain from
`run_simulation` down to that computation was traced rather than assumed:
`simulation.py:333-338`'s `run_simulation(...)` call
(`.venv/Lib/site-packages/flwr/simulation/run_simulation.py`) drives the
SuperLink/Fleet VCE, whose Ray backend is
`RayBackend` (`.venv/Lib/site-packages/flwr/server/superlink/fleet/vce/backend/raybackend.py:39-40`).
`RayBackend.build()` constructs `self.pool = BasicActorPool(actor_type=ClientAppActor,
client_resources=self.client_resources, ...)` (`raybackend.py:137-141`), and
`BasicActorPool.__init__` (imported into `raybackend.py:30` from
`flwr/simulation/ray_transport/ray_actor.py:403-428`) calls
`self.actors_capacity = pool_size_from_resources(client_resources)`
(`ray_actor.py:428`) — the function that actually runs the arithmetic below
(`ray_actor.py:86-134`):

```python
num_cpus = node_resources.get("CPU", 0)
num_gpus = node_resources.get("GPU", 0)
num_actors = int(num_cpus / client_resources["num_cpus"])
if client_resources["num_gpus"] > 0.0:
    if num_gpus:
        num_actors = min(num_actors, int(num_gpus / client_resources["num_gpus"]))
    else:
        num_actors = 0
```

With `num_cpus=1` per client actor against a dual-socket host (`docs/agents/execution-model.md:23`:
"dual Xeon Platinum 8276" — dozens of physical cores; exact core count is a
public CPU spec, not something logged in this repo, so not asserted precisely
here), the CPU term is essentially never binding. The GPU term is: Ray detects
one physical GPU on this host (`docs/agents/execution-model.md:23`: "A100 40GB
PCIe", singular), so `num_gpus` (Ray's resource units) is 1.0, and

```
num_actors = floor(1.0 / client_gpus)
```

which is exactly `2` at `client_gpus=0.5`, `4` at `0.25`, `8` at `0.125` — the
groundwork doc's stated figures, now traced to Flower's own scheduling code
rather than asserted. **This is the entire mechanism behind "concurrency" in
this pipeline: the number of Ray client actors Flower's `ActorPool` is allowed
to hold, gated by how many `client_gpus`-sized fractional-GPU slots fit in one
detected GPU.** It says nothing on its own about whether the GPU can actually
execute that many training jobs in parallel without slowdown — that is
Question 1/3, unmeasured (see below).

---

## 5. The bottleneck question (Question 3) — reasoned from code, not measured

The ticket rules host-level live VRAM/RAM telemetry out of scope to source
directly, and no probe at `client_gpus != 0.5` has been run (Section 2), so
nothing here is an empirical finding. This section only narrows the
*candidates*, from what is already logged or configured:

- **GPU compute (SM occupancy), not GPU memory, is the most likely limiter,
  and it is untested.** ADR-0013 (`docs/adr/0013-execution-infrastructure-failures.md:22-25`)
  measured ~5.0 GB of this account's VRAM at `client_gpus=0.5` (2 concurrent
  actors) against a 40 GB card with a co-tenant holding ~11 GB — roughly 24 GB
  free. That rules out VRAM *capacity* as a near-term ceiling at 0.5, but says
  nothing about compute throughput: two MobileNetV2GN training actors sharing
  one A100's streaming multiprocessors can already saturate compute occupancy
  well before VRAM fills, in which case going to 4 or 8 actors would just
  time-slice the same silicon and wall time would stop improving — a real
  ceiling, not a broken measurement, but one that requires the Part B probes
  to see because no per-round GPU-utilization telemetry is captured anywhere
  in `src/fedmaq/core/telemetry.py` (confirmed: no `nvidia-smi` or
  `torch.cuda.utilization` call anywhere in that file or in `simulation.py`).
- **CPU is unlikely to bind.** `num_cpus=1` per actor (`simulation.py:322`)
  against a dual-Xeon host means even 8 actors at `client_gpus=0.125` ask for 8
  cores on a host with far more than that available — the pool-size formula in
  Section 4b would need `num_gpus` to be the limiting term long before CPU
  count is, for any of the three settings this ticket tests.
- **System RAM / Ray object-store pressure is the one bottleneck already
  flagged explicitly in code, as a failure mode rather than a throughput
  curve.** `scripts/run_matrix.py:156-159`'s `--run_timeout_seconds` help text:
  "Ray can deadlock waiting on an actor that never starts, which is the
  documented low-system-RAM failure mode." `docs/agents/execution-model.md:56-58`
  gives the concrete numbers: Ray sizes its object store at ~30% of *total*
  RAM by default (~19 GB on this 64 GB host, `docs/agents/execution-model.md:23`),
  against ~39 GB actually free once the sweep's own footprint is accounted
  for — which is why every dispatched sweep already overrides
  `ray.object_store_gb=4` (`execution-model.md:51`). This is a documented
  *cliff* (deadlock), not a graceful degradation, so it is plausibly the
  binding constraint on how far concurrency can be pushed even if GPU compute
  has headroom — but whether it actually triggers at `client_gpus=0.25` or
  `0.125` is exactly what the groundwork doc's Part B Step 2 sustained-run RAM
  sampler is designed to check, and that has not been run either.
- **Network/I/O is not a real host bottleneck.** `bandwidth_mbps=10.0` (`conf/experiment/default.yaml:20`)
  is a simulated Pi5/WiFi parameter consumed only by the cost model
  (`telemetry.py`'s `client_round_delay`, Section 3 above) — it does not throttle
  any real data transfer. Real inter-actor communication in a Ray-simulated
  Flower run goes through Ray's local object store (shared memory / local
  disk spill), not a network link, so I/O contention, if any, would show up as
  object-store spill warnings (which Part B Step 1's log-capture already asks
  for) rather than as network latency.

**In short: GPU compute occupancy and system-RAM/object-store pressure are the
two code-grounded candidates; CPU count and simulated network bandwidth are
not real candidates for the host bottleneck.** Distinguishing GPU-compute from
RAM as the actual ceiling requires the measurements the ticket already asks
for and that have not yet been run.

---

## 6. Where this leaves the ticket's "already settles" section

| Claim | Status after this pass |
| --- | --- |
| Total elapsed 109h, median 0.57h, max 1.58h | **Confirmed, independently reproduced from raw telemetry in this pass** (Section 1: 109.0707h / 0.5657h / 1.5754h, aggregated fresh from all 177 `experiment_log*.csv` files, not quoted from the ticket or from `docs/experiments/issue-31-concurrency-ceiling.md`). |
| `system/cumulative_time_sec` sums to ~1055h against 109h, "~10x concurrency the harness is already extracting" | **Contradicted, independently confirmed.** Traced directly to `telemetry.py:291-295` and `conf/experiment/default.yaml:20-21`: the 1055h figure is a synthetic Raspberry-Pi-5/WiFi deployment-time projection, unrelated to how many Ray actors ran concurrently on the host. All 177 manifests used one `client_gpus` value, so there is no cross-setting wall-time comparison to draw a concurrency ratio from even in principle. This matches the correction already recorded in `docs/experiments/issue-31-concurrency-ceiling.md`, reached here independently from the source rather than from that doc's prose. |
| Client-side ~98-100% of compute, server ~0-2.3% | Not re-verified numerically in this pass; independently confirmed to be a ratio *within* the same synthetic cost model (Section 3), so it is a valid, self-consistent finding regardless of the wall-time correction above. |
| `*_sim_time_sec` telemetry is simulated by design (ADR-0002), only `wall_time_sec` is real | **Confirmed directly from source.** `telemetry.py:291-295` (`*_sim_time_sec` from the cost model) vs. `telemetry.py:309-311` (`wall_time_sec` from `time.perf_counter()`). |

**Could not be verified in this pass (requires runs, not code reading):** the
per-host throughput ceiling itself, and which of GPU-compute-occupancy vs.
system-RAM/object-store pressure actually binds first. Both remain exactly
where the existing groundwork doc (`docs/experiments/issue-31-concurrency-ceiling.md`,
Parts B and C) left them — commands prepared, no results returned yet.
