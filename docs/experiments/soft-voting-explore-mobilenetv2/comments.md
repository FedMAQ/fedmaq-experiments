# Soft-Voting Explore Sweep — Analysis (Priority 1, Pass 1)

Full decision-rule reasoning: `docs/DECISIONS.md` 2026-07-18 entries 33-35.

## Decision rule gap

Decision 30 (2026-07-18) mandated a noise-margin decision rule but never sourced a numeric margin — single-seed exploration runs have no within-cell variance, and no repeated-seed characterization existed for MobileNetV2GN at the time Pass 1 launched. Rather than block on a full multi-seed characterization pass upfront, the empirical spread of the sweep grid itself was used as a stand-in noise floor for this pass: 14/16 cells cluster within a ~1.6pp band, which was treated as "noise-scale" separation.

## Reading the sweep

- **entropy_weight=2.0, precision_weight=0.5** (idx10, 0.5179) clears the empirical band by +1.6-3.4pp — the only cell that does. Tentatively adopted as the Pass 1 pick.
- **soft_voting on vs off** (+1.24pp) sits inside the band — not distinguishable from noise at this sample size. Soft-voting is *not* dropped (idx10 requires it), but the ablation alone doesn't independently justify it.
- **ew=4.0, pw=4.0** (idx17, 0.4601) is the sweep floor, but also the run that needed 5 Ray-crash relaunches — its provenance is rockier than the other 17 runs (post-hoc CSV dedup of duplicate rows). Not excluded, but not to be read as a clean "high ew+pw hurts" signal without a rerun.
- No obvious monotonic trend in either ew or pw alone — the surface looks like ew=2.0/pw=0.5 is a genuine local optimum rather than an edge of a monotone gradient, but this is a single-seed read and could be noise-driven curvature.

## Status

**Provisional, not frozen.** User has agreed to a future multi-seed re-verification pass (repeat idx10 plus 1-2 other cells, e.g. grid center and the sv ablation pair, with 2-3 additional seeds) before this feeds the confirmatory grid. Until then, `entropy_weight=2.0`, `precision_weight=0.5`, `soft_voting=true` is the working assumption carried into Pass 2 (capacity-EMA, grad-norm smoothing, client-KD-reg), which is largely orthogonal to these settings per Decision 29.
