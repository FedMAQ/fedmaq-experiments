# ADR-0005 — The baseline stack: membership, exclusions, and validation scope

**Status**: Accepted · 2026-07-16 through 2026-07-24
**Supersedes**: `docs/DECISIONS.md` Decisions 21–26, 45, 49 (file deleted; see ADR-0014)

## Context

The thesis proposal named eight baselines. Two were dropped after implementation
proved them infeasible or structurally broken at this experiment's scale, and one
(FedKD) needed two corrections before it was usable at all. This ADR records the
surviving stack, why the two exclusions are structural rather than provisional, and
how much validation each surviving baseline owes.

## Decision

### The stack

| Group | Algorithms |
| --- | --- |
| Seminal controls | FedAvg, FedProx |
| Pure quantization | FedPAQ, DAdaQuant |
| Pure KD | FedDistill *(FedMD dropped)* |
| Hybrid Q+KD | FedKD *(CFD dropped)* |

Six baselines plus FedMAQ. Dropped code is **retained, not deleted** —
`conf/algorithm/{fedmd,cfd}.yaml` and their hooks stay for reproducibility, marked
dropped in every registry, and excluded from all sweeps.

### FedMD is dropped: infeasible pretrain cost

FedMD's one-time 20-epoch (10 public + 10 private) transfer-learning pretrain *per
client*, combined with `client_gpus=1.0` forcing serial Ray actor execution, meant
~90 distinct clients each paid that cost sequentially. Wall-clock tracked
7–8.5 min/round even after trimming the digest phase from 5 epochs to 3, projecting
6+ hours for a single 50-round smoke arm — infeasible to reproduce across a 3-seed
grid. A convergence-based pretrain-stop was scoped and rejected: its cap of 100 can
run *longer* than the fixed 10/10 in the worst case, directly fighting the
feasibility goal, and the plateau-detection path itself needs implementation and
validation — the exact overhead being escaped. Keeping FedMD via a higher fixed cap
or a FedMD-only reduced client count was also rejected; both still require
defending a heavy, paper-deviating baseline.

FedDistill remains the sole pure-KD baseline. It is also prediction/logit-based and
covers the same "no full-weight-sharing" comparison axis at a fraction of the cost.

**FedMD is additionally excluded from `scripts/golden_diff.py`'s default
`GOLDEN_SET` and from smoke matrices.** It stayed in the Step 2 golden set purely
for regression coverage of that migration. It is by far the slowest config —
disk-persisted multi-phase training, up to 4× `run_epochs` per round, ~9 minutes
for a 2-round `experiment=ci` config against under a minute for most baselines.
Re-add it only for a change that actually touches its code path.

### CFD is dropped: structural collapse at production client count

CFD collapses to chance accuracy at both skews. Three isolated repro runs located
the cause, and it is **not** the client-side codec the original audit suspected —
that code was probed directly and is correct.

1. **Server-side dual distillation is exonerated.** A 5-client full-participation
   discriminator run showed the server model tracks its targets correctly once
   given enough gradient steps. An earlier same-session reading calling this a
   server-side mode collapse was a toy artifact (100 public samples ÷ batch 64 = 2
   gradient steps/round — undertraining, not a bug) and is retracted.
2. **The defect is upstream, at scale.** At 50 clients / `client_fraction=0.1`,
   `targets_acc` pins near chance with individual clients one-hot-voting the *same
   single class* for all public samples from round 1. Each client's partition is
   ~470 samples at 100 clients on CIFAR-10, and 5 local CE epochs from a fresh init
   overfits to 1–2 dominant local classes — healthy local train accuracy (50–65%)
   coexists with near-random generalization to the disjoint, class-balanced public
   set. CFD's 1-bit `b_up` then forces each vote to full commitment to that one
   class with zero hedged signal, unlike the other KD baselines' temperature-scaled
   soft-probability averaging, so a few overfit voters dominate the round outright.
3. **Raising the vote bit-width does not rescue it.** At `b_up=b_down=4`,
   `targets_acc` barely moved (14→21%) because the underlying *prediction* is
   wrong, not merely imprecisely encoded. This rules out bit-width as a
   config-only fix.

Rejected mitigations: raising `client_fraction` to dilute degenerate votes (clients
still individually overfit regardless of how many are sampled); CFD-only local-epoch
reductions, regularization, or skipping round 1's contribution — all touch shared
hyperparameters or need their own validation pass, and none address the mismatch
between CIFAR-10's per-client data budget at 100 clients and a 1-bit hard-vote
protocol. CFD becomes an exploration-appendix note: the collapse is real and
diagnosed, not a baseline result.

### FedKD needed two corrections before it was usable

**An SVD rank floor.** Energy→rank was non-monotonic on concentrated
(depthwise-separable) spectra: retained rank could collapse toward 1 even as the
round-scheduled energy target rose, starving the convergence-critical window. Both
candidate fixes were probed against the production code path with real deltas over
15 simulated rounds — raising `tmin` alone still dipped non-monotonically
mid-schedule, while a minimum-rank floor (`retained rank ≥ min_rank_frac ×
full_rank`) eliminated the dip. Landed as `min_rank_frac` on `compress_tensor`,
threaded through both compression hooks, defaulting to `0.25`.

**A compact student on the same backbone.** The iso-architecture switch (ADR-0004)
shrank the full model to ~2.24M, leaving FedKD's old SimpleCNN student (~2.16M)
neither meaningfully smaller nor on the depthwise-separable family the rest of the
grid trains — a comparison confound and a collapsed compact-student story. Rejected:
an iso-architecture student=teacher pairing (degenerate mutual distillation between
identical-capacity nets, unfaithful to the source's mentor-mentee asymmetry) and
keeping SimpleCNN while merely documenting the confound. Chosen: a **width-0.5
MobileNetV2GN student** (~0.59M on CIFAR-10, ~0.26× the full model) — genuinely
smaller, same backbone, so the SVD now compresses depthwise-separable deltas and
the compact-student story holds. Scoped to CIFAR; FEMNIST keeps its TinyCNN student.

**The residual gap is an open finding, not a closed investigation.** The
rank-starvation fix is confirmed working — a 50-round re-run on the new student
reached 30.10% (α=0.1) and 38.31% (α=1.0) peak, well above chance, with
`mean_rank_retained` held at its floor throughout. But sitting *at* the floor
rather than climbing past it means "SVD is too lossy for depthwise-separable
weights" remains the leading explanation for the remaining 15–27pp gap against
every other baseline. FedKD's absolute level is architectural, not a tuning
failure (ADR-0011) — do not read it as a bug.

Pre-fix FedKD numbers were measured on the retired SimpleCNN student and are
architecture-confounded; **do not compare them against post-fix figures.** The
causal evidence for the fix is the same-model A/B, not a pre/post delta across the
student swap.

### Validation scope: anchor plus directional, not full replication

Reproducing every baseline's original paper setup exactly is expensive, and many
source papers don't publish enough detail for it anyway. Doing this for all six
would be disproportionate to a thesis and risks becoming its own side project.

**Full-fidelity replication against published numbers is owed for one or two
well-documented anchor baselines only** — FedAvg on CIFAR-10 is the natural
candidate. Every other baseline is validated *directionally* against its paper's
key claims rather than absolute numbers: that FedPAQ/DAdaQuant reduce
communication with roughly the reported order-of-magnitude accuracy tradeoff, that
relative orderings hold, and so on.

The goal is confidence that each implementation is *correct enough* to be a fair
comparison point, not a standalone reproducibility study. Revisit this scope only
if a baseline's correctness becomes genuinely contested — results contradicting its
paper's qualitative claim, not merely its absolute numbers.

## Consequences

- When validating a baseline, check absolute numbers only for the FedAvg/CIFAR-10
  anchor; for the rest, check the qualitative claim direction.
- FedMD and CFD code paths are parked, not maintained. A refactor that touches
  shared seams should keep them compiling and bit-exact (ADR-0006) but need not
  optimize or extend them.
- The baseline status table lives in `.claude/rules/experiment-design.md`, which
  is where an agent adding or porting a baseline updates status.
